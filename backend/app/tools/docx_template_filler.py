"""通用 DOCX 模版填充器（跨 run 版）。

因为示例数据在模版里常被拆成多个 <w:t> run（如 杭州市/泽熙信息/科技有限公司），
需要在 run 元素级别做替换，单纯字符串替换会遗漏大量场景。
"""

import re
import tempfile
import zipfile
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "data" / "jianlimoban"

PLACEHOLDER = "[待填写]"

WT_PATTERN = re.compile(r"(<w:t[^>]*>)([^<]*)(</w:t>)")


def _merged_experiences(collected: dict) -> list[dict]:
    """把 work/internship/projects 合并成有序列表。"""
    exps = []
    for e in collected.get("work_experience") or []:
        exps.append({"type": "工作", **e})
    for e in collected.get("internship") or []:
        exps.append({"type": "实习", **e})
    for e in collected.get("projects") or []:
        exps.append(
            {
                "type": "项目",
                "company": e.get("name", ""),
                "title": e.get("role", ""),
                **e,
            }
        )
    return exps


def _extract_texts(xml: str) -> tuple[list[str], list[tuple]]:
    """提取所有 <w:t> 的文本内容，返回 (texts, match_spans)。"""
    matches = list(WT_PATTERN.finditer(xml))
    texts = [m.group(2) for m in matches]
    spans = [(m.start(), m.end(), m.group(1), m.group(3)) for m in matches]
    return texts, spans


def _rebuild_xml(xml: str, spans: list[tuple], new_texts: list[str]) -> str:
    """根据新 texts 列表重新拼接 XML。"""
    result = []
    last_end = 0
    for (start, end, prefix, suffix), new_text in zip(spans, new_texts):
        result.append(xml[last_end:start])
        result.append(prefix + new_text + suffix)
        last_end = end
    result.append(xml[last_end:])
    return "".join(result)


def _replace_sequence(
    texts: list[str],
    new_texts: list[str],
    target_seq: list[str],
    replacement_seq: list[str],
    match_mode: str = "equal",
):
    """在 texts 里找连续出现的 target_seq，替换为 replacement_seq。

    match_mode:
      - "equal": 完全匹配
      - "contains": texts[i] 包含 target_seq[i]
      - "startswith": texts[i] 以 target_seq[i] 开头
    """
    n = len(target_seq)
    i = 0
    while i <= len(texts) - n:
        hit = True
        for k in range(n):
            t = new_texts[i + k]
            pat = target_seq[k]
            if match_mode == "equal":
                ok = t == pat
            elif match_mode == "contains":
                ok = pat in t
            elif match_mode == "startswith":
                ok = t.startswith(pat)
            else:
                ok = False
            if not ok:
                hit = False
                break
        if hit:
            for k in range(n):
                if k < len(replacement_seq):
                    new_texts[i + k] = replacement_seq[k]
            i += n
        else:
            i += 1


def _process_xml(xml: str, collected: dict) -> str:
    """核心处理：基于 w:t 元素级的替换。"""
    basic = collected.get("basic_info") or {}
    edu = (collected.get("education") or [{}])[0]
    exps = _merged_experiences(collected)
    skills = collected.get("skills") or []
    summary = collected.get("personal_summary") or ""

    name = basic.get("name") or PLACEHOLDER
    phone = basic.get("phone") or ""
    email = basic.get("email") or ""
    location = basic.get("location") or ""

    school = edu.get("school") or ""
    major = edu.get("major") or ""
    degree = edu.get("degree") or "本科"
    courses = edu.get("courses") or ""

    def exp_val(idx, key, fallback=""):
        if idx < len(exps):
            return exps[idx].get(key) or fallback
        return fallback

    def skill_val(i, fallback=""):
        return skills[i] if i < len(skills) else fallback

    texts, spans = _extract_texts(xml)
    new_texts = texts[:]

    # ── 跨 run 模式替换 ─────────────────────────────
    # 用 contains 模式匹配，兼容带尾随空格的 run（如"大学         "）
    _replace_sequence(
        texts,
        new_texts,
        ["哆啦缘梦科技", "大学"],
        [school, ""],
        match_mode="contains",
    )
    _replace_sequence(
        texts,
        new_texts,
        ["哆啦缘梦", "科技", "大学"],
        [school, "", ""],
        match_mode="contains",
    )
    _replace_sequence(
        texts,
        new_texts,
        ["鹿大仙设计", "大学"],
        [school, ""],
        match_mode="contains",
    )
    _replace_sequence(
        texts,
        new_texts,
        ["鹿大仙", "设计", "大学"],
        [school, "", ""],
        match_mode="contains",
    )
    _replace_sequence(
        texts,
        new_texts,
        ["XX", "设计大学"],
        [school, ""],
        match_mode="contains",
    )

    # 公司名拆分：支持 "杭州市"/"  杭州市" + "泽熙信息" + "科技有限公司..."
    # 模版常有双副本（Word 兼容模式），先扫描总出现次数，按段分配
    _company_positions = []
    i = 0
    while i < len(texts) - 2:
        t0 = new_texts[i].strip()
        t1 = new_texts[i + 1]
        t2 = new_texts[i + 2]
        if (
            t0 == "杭州市"
            and "泽熙信息" in t1
            and t2.lstrip().startswith("科技有限公司")
        ):
            _company_positions.append(i)
            i += 3
        else:
            i += 1

    # 按职位示例分组判定"段数"：示例里 [校园大使主席, 市场营销（实习生）, 软件工程师]
    # 数出现的不同职位种数，作为段数；每段分配给 exps 里的一个
    segment_titles = ["校园大使主席", "市场营销（实习生）", "软件工程师"]
    segment_found = []
    for pos in _company_positions:
        # 往后找 3 个 run 看包含哪个示例职位
        matched_title = None
        for j in range(pos, min(pos + 4, len(texts))):
            for t in segment_titles:
                if t in new_texts[j]:
                    matched_title = t
                    break
            if matched_title:
                break
        segment_found.append(matched_title or "unknown")

    # 建立：示例职位 → exp_idx 的映射（按出现顺序）
    unique_seg_order = []
    for s in segment_found:
        if s not in unique_seg_order:
            unique_seg_order.append(s)

    # 替换
    for pos, seg in zip(_company_positions, segment_found):
        if seg in unique_seg_order:
            exp_idx = unique_seg_order.index(seg)
        else:
            exp_idx = 0
        co = exp_val(exp_idx, "company") or ""
        new_texts[pos] = co
        new_texts[pos + 1] = ""
        tail = new_texts[pos + 2]
        m = re.match(r"^\s*科技有限公司\s*", tail)
        if m:
            new_texts[pos + 2] = tail[m.end() :]

    # 没有"泽熙信息"的变体：杭州市 + 科技有限公司...
    i = 0
    while i < len(texts) - 1:
        t0 = new_texts[i].strip()
        t1 = new_texts[i + 1]
        if t0 == "杭州市" and t1.lstrip().startswith("科技有限公司"):
            co = exp_val(company_occurrence, "company") or ""
            new_texts[i] = co
            tail = t1
            m = re.match(r"^\s*科技有限公司\s*", tail)
            if m:
                new_texts[i + 1] = tail[m.end() :]
            company_occurrence += 1
            i += 2
        else:
            i += 1

    # 姓名：学校替换后残留的"哆啦缘梦"单 run 即为姓名位置
    for i, t in enumerate(new_texts):
        if t == "哆啦缘梦":
            new_texts[i] = name

    # 姓名：学校替换后残留的"哆啦缘梦"单 run 即为姓名位置
    for i, t in enumerate(new_texts):
        if t == "哆啦缘梦":
            new_texts[i] = name

    # 3. 出生年月 2025.05 这种拆分：20xx(已替为2025) + .05
    #    用户没出生年月 → 把 "出生年月：" 后面的日期清空
    for i, t in enumerate(texts):
        if t == "出生年月：":
            # 找后面 2 个 run 清空
            for j in range(i + 1, min(i + 4, len(texts))):
                if re.match(r"^\d+$", new_texts[j]) or re.match(
                    r"^\.\d+$", new_texts[j]
                ):
                    new_texts[j] = ""

    # ── 单 run 字符串替换 ───────────────────────────
    # 注意顺序：先长的、后短的
    single_run_reps = [
        # 长主修课程
        (
            "管理学、微观经济学、宏观经济学、管理信息系统、统计学、会计学、财务管理、市场营销、经济法、消费者行为学、国际市场营销",
            courses or "",
        ),
        (
            "金融学、管理学、微观经济学、宏观经济学、中级微观经济学、中级宏观经济学、统计学、会计学、中级财务会计、税法、审计、税收筹划、国家预算、外汇理论与实务、私募股权投资与理论、计算机应用等。",
            courses or "",
        ),
        # 长自我评价
        (
            "深度互联网从业人员，对互联网保持高度的敏感性和关注度，熟悉产品开发流程，有很强的产品规划、需求分析、交互设计能力，能独立承担APP和WEB项目的管控工作，善于沟通，贴近用户。",
            summary or "",
        ),
        # 技能证书
        ("普通话一级甲等；", (skill_val(0) + "；") if skill_val(0) else ""),
        (
            "大学英语四/六级（CET-4/6），良好的听说读写能力，快速浏览英语专业文件及书籍；",
            (skill_val(1) + "；") if skill_val(1) else "",
        ),
        (
            "通过全国计算机二级考试，熟练运用office相关软件。",
            (skill_val(2) + "。") if skill_val(2) else "",
        ),
        # 经历 bullet 描述
        (
            "目标带领自己的团队，辅助泽熙公司完成在各高校的“伏龙计划”，向全球顶尖的",
            exp_val(0, "description"),
        ),
        (
            "整体运营前期开展了相关的线上线下宣传活动，中期为进行咨询的人员提供讲解。后期进行了项目的维护阶段，保证了整个项目的完整性。",
            "",
        ),
        (
            "带领本校团队超额完成鹿大仙设计公司的业绩，绩效占到大连区的30%左右，是大连区绩效的重要组成部分，同时推动了东北地区业绩的完成。",
            "",
        ),
        (
            "负责公司线上端资源的销售工作（以开拓客户为主），公司主要资源以广点通、智汇推、百度、小米、",
            exp_val(1, "description"),
        ),
        (
            "实时了解行业的变化，跟踪客户的详细数据，为客户制定更完善的投放计划（合作过珍爱网、世纪佳缘、",
            "",
        ),
        (
            "负责公司业务系统的设计及改进，参与公司网上商城系统产品功能设计及实施工作。",
            exp_val(2, "description"),
        ),
        (
            "负责客户调研、客户需求分析、方案写作等工作， 参与公司多个大型电子商务项目的策划工作，担任大商集团网上商城一期建设项目经理。",
            "",
        ),
        # 公司名示例（整 run 内）
        ("杭州市泽熙信息科技有限公司", exp_val(1, "company")),
        ("广州鹿大仙设计信息科技有限公司", exp_val(2, "company")),
        ("鹿大仙设计信息科技有限公司", exp_val(2, "company")),
        ("泽熙信息科技有限公司", exp_val(1, "company")),
        # 职位（整 run 内）
        ("市场营销（实习生）", exp_val(1, "title")),
        ("软件工程师", exp_val(2, "title")),
        # 学校/专业 整 run
        ("市场营销（本科）", f"{major}（{degree}）" if major else ""),
        ("税务专业（本科）", f"{major}（{degree}）" if major else ""),
        # 姓名（放后面，避免误伤"哆啦缘梦科技"）
        ("胡小豆", name),
        # 固定文本
        ("金融公司推送实习生资源。", " 配置文件。"),
        ("、沃门户等；", ""),
        ("视频、京东等客户）", ""),
        ("AXA金融公司", ""),
        ("AXA", ""),
        ("校园大使主席", exp_val(0, "title")),
        # 基本信息
        ("13888888888", phone),
        ("135xxxxxxxxx", phone),
        ("13500000000", phone),
        ("888888@163.com", email),
        ("00000@xx.com", email),
        ("0000@xx.me", email),
        ("浙江省杭州市滨江区", location),
        ("广东省广州市海珠区滨江东路", location),
        ("广东省珠海市", location),
        # 杂项（用户没提供就清空）
        ("1996.05", ""),
        ("167cm", ""),
        ("汉", "" if not basic.get("ethnicity") else basic.get("ethnicity")),
        # 日期占位
        ("20xx", "2025"),
        ("20XX", "2025"),
    ]

    # 先按 src 长度降序，避免短字符串破坏长字符串
    single_run_reps.sort(key=lambda p: -len(p[0]))

    for i, t in enumerate(new_texts):
        for src, dst in single_run_reps:
            if src and src in t:
                new_texts[i] = new_texts[i].replace(src, dst)

    # 清理："哈尔滨商业大学" 紧跟的 "大学" 这类残留
    if school and school.endswith(("大学", "学院")):
        for i, t in enumerate(new_texts):
            if t == school and i + 1 < len(new_texts):
                if new_texts[i + 1].strip() in ("大学", "学院"):
                    new_texts[i + 1] = ""

    # 清理："出生年月：" "身    高：" "政治面貌" "民    族" 这些用户没提供的字段
    # 这些整行删除放在 B 类，A 类只把值清空
    if not basic.get("birth"):
        for i, t in enumerate(new_texts):
            if t in ("出生年月：", "出生年月") and i + 1 < len(new_texts):
                # 后 2 个 run 如果是日期数字就清空
                for j in range(i + 1, min(i + 3, len(new_texts))):
                    if re.match(r"^\d{2,4}$|^\.\d+$", new_texts[j].strip()):
                        new_texts[j] = ""

    if not basic.get("height"):
        for i, t in enumerate(new_texts):
            if "身" in t and "高" in t and "：" in t:
                # 值在本 run 或下一 run
                after = t.split("：", 1)[-1]
                if re.match(r"^\d+cm?$|^$", after.strip()):
                    new_texts[i] = t.split("：")[0] + "："

    if not basic.get("political"):
        for i, t in enumerate(new_texts):
            if t.startswith("政治面貌") and "：" in t:
                new_texts[i] = t.split("：")[0] + "："

    if not basic.get("ethnicity"):
        for i, t in enumerate(new_texts):
            if t == "：汉" or t == "汉":
                new_texts[i] = "："

    return _rebuild_xml(xml, spans, new_texts)


def fill_template_bytes(
    template_path: Path, collected_info: dict, resume_draft: str = ""
) -> bytes:
    """读取模版 → 替换 → 返回填好的 docx bytes。"""
    if not template_path.exists():
        raise FileNotFoundError(f"模版不存在: {template_path}")

    with zipfile.ZipFile(template_path, "r") as z:
        files = {n: z.read(n) for n in z.namelist()}

    xml = files["word/document.xml"].decode("utf-8")
    xml = _process_xml(xml, collected_info)
    files["word/document.xml"] = xml.encode("utf-8")

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    with zipfile.ZipFile(out.name, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    out.close()

    with open(out.name, "rb") as f:
        result = f.read()
    Path(out.name).unlink(missing_ok=True)
    return result


def list_templates() -> list[str]:
    if not TEMPLATE_DIR.exists():
        return []
    return sorted(
        f.name for f in TEMPLATE_DIR.glob("*.docx") if not f.name.startswith("_")
    )


def get_template_path(name: str | None = None) -> Path:
    templates = list_templates()
    if not templates:
        raise FileNotFoundError(f"模版目录为空: {TEMPLATE_DIR}")
    if name is None:
        return TEMPLATE_DIR / templates[0]
    for t in templates:
        if t == name or t.startswith(name):
            return TEMPLATE_DIR / t
    raise FileNotFoundError(f"未找到模版: {name}")
