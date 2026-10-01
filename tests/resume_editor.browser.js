// Optional local browser regression: playwright-cli run-code --filename=tests/resume_editor.browser.js
async (page) => {
  const checks = [];
  const check = (condition, name) => {
    if (!condition) throw new Error(name);
    checks.push(name);
  };
  const input = (path) => page.locator(`[data-input-path="${path}"]`);
  const preview = (path) => page.frameLocator('#preview-frame').locator(`[data-field-path="${path}"]`);
  const settled = () => page.waitForFunction(() => document.getElementById('render-status').textContent === '已实时更新');
  const templates = page.locator('.tpl-item');
  await page.setViewportSize({width: 1440, height: 960});
  await page.goto('http://127.0.0.1:8765/resume');
  await settled();
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.mouse.move(10, 10);
  check(await input('/basic_info/name').count() === 1, '表单加载');
  check(await preview('/sections/2/title').textContent() === '项目经历', '默认栏目标题');

  // Playwright OOPIF coordinates omit ancestor CSS transforms. Use the visible
  // iframe bounds and the observed child rect for actual mouse hit testing.
  const point = await preview('/basic_info/name').evaluate((node) => {
    const rect = node.getBoundingClientRect();
    return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
  });
  const frameBox = await page.locator('#preview-frame').boundingBox();
  const scale = frameBox.width / 794;
  await page.mouse.move(frameBox.x + point.x * scale, frameBox.y + point.y * scale);
  await page.locator('.field.linked').waitFor();
  check(await page.locator('.field.linked label').textContent() === '姓名', '预览悬停高亮对应输入框');
  check(await preview('/basic_info/name').evaluate((node) => getComputedStyle(node).outlineStyle) === 'dashed', '预览文字虚线框');
  await page.mouse.click(frameBox.x + point.x * scale, frameBox.y + point.y * scale);
  await page.waitForFunction(() => document.activeElement.dataset.inputPath === '/basic_info/name');
  check(true, '点击预览定位输入框');
  await input('/basic_info/name').fill('实时编辑验证');
  await preview('/basic_info/name').filter({hasText: '实时编辑验证'}).waitFor();
  await settled();
  check(await preview('/basic_info/name').textContent() === '实时编辑验证', '输入即时更新及后续渲染');

  let slowStarted;
  const slowRequest = new Promise((resolve) => { slowStarted = resolve; });
  await page.route('**/api/resume/render/*?interactive=true', async (route) => {
    const data = route.request().postDataJSON();
    if (data.basic_info.name === '旧值') {
      slowStarted();
      await page.waitForTimeout(350);
    }
    try { await route.continue(); } catch (_) { /* Aborted stale request is expected. */ }
  });
  await input('/basic_info/name').fill('旧值');
  await slowRequest;
  await input('/basic_info/name').fill('最新值');
  await templates.nth(1).click();
  await settled();
  await page.waitForTimeout(400);
  check(await preview('/basic_info/name').first().textContent() === '最新值', '慢请求与切换模板保留最新输入');
  await page.unroute('**/api/resume/render/*?interactive=true');

  await page.getByRole('button', {name: '＋ 添加工作 / 实习经历', exact: true}).click();
  await input('/work_experience/1/company').fill('新公司');
  await input('/work_experience/1/bullets/0').fill('新经历要点');
  const workSection = page.locator('.edit-section').filter({has: page.locator('.section-caption', {hasText: /^实习经历$/})});
  await workSection.getByRole('button', {name: '删除', exact: true}).first().click();
  await input('/work_experience/0/company').fill('删除首项后的新公司');
  await settled();
  check(await preview('/work_experience/0/company').textContent() === '删除首项后的新公司', '增删后索引重新对应');
  check(await preview('/work_experience/0/bullets/0').textContent() === '新经历要点', '增删后保留正确内容');

  await page.getByRole('button', {name: '＋ 添加项目', exact: true}).click();
  const projectsSection = page.locator('.edit-section').filter({has: page.locator('.section-caption', {hasText: /^项目经历$/})});
  await projectsSection.getByRole('button', {name: '改为分条要点', exact: true}).last().click();
  await input('/projects/1/polished_bullets/0').fill('新项目第一条');
  await projectsSection.getByRole('button', {name: '改为整段描述', exact: true}).last().click();
  check(await input('/projects/1/description').inputValue() === '新项目第一条', '分条与整段切换保留内容');
  await projectsSection.getByRole('button', {name: '改为分条要点', exact: true}).first().click();
  await input('/projects/0/polished_bullets/0').fill('替换后的要点');
  await projectsSection.getByRole('button', {name: '删除项目要点 1', exact: true}).first().click();
  check(await input('/projects/0/description').inputValue() === '', '删除最后项目要点不复活旧描述');

  await page.getByRole('button', {name: '＋ 添加技能分组', exact: true}).click();
  await page.getByRole('textbox', {name: '分组名称', exact: true}).last().fill('技术/~栈');
  await input('/skill_groups/技术~1~0栈/0').fill('Python');
  await settled();
  check(await preview('/skill_groups/技术~1~0栈/0').textContent() === 'Python', '分组重命名及特殊字符路径');
  await page.getByRole('button', {name: '删除分组', exact: true}).first().click();
  await page.getByRole('button', {name: '删除分组', exact: true}).first().click();
  await settled();
  check(await page.locator('[data-input-path^="/skills/"]').count() === 0, '删除最后分组不复活旧技能');

  // Clicking export before the debounce must still send the current value.
  await input('/basic_info/name').fill('导出即时值');
  await page.locator('#btn-export').click();
  const exportRequest = page.waitForRequest((request) => request.url().includes('/api/resume/export/docx/'));
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', {name: '通用 Word', exact: true}).click();
  const request = await exportRequest;
  const download = await downloadPromise;
  check(request.postDataJSON().basic_info.name === '导出即时值', '导出使用最新输入');
  check(download.suggestedFilename().includes('导出即时值'), 'Word 下载成功');
  const htmlResponse = await page.request.post('http://127.0.0.1:8765/api/resume/render/t002-jianyue', {data: request.postDataJSON()});
  const exportedHtml = await htmlResponse.text();
  check(htmlResponse.ok() && !exportedHtml.includes('<script') && !exportedHtml.includes('resume-field-active'), '打印HTML无交互脚本或高亮');

  for (let index = 0; index < 6; index += 1) {
    await templates.nth(index).click();
    await settled();
    check(await preview('/basic_info/name').first().textContent() === '导出即时值', `模板 ${index + 1} 内容一致`);
  }
  for (const width of [1440, 1024, 768, 390]) {
    await page.setViewportSize({width, height: 844});
    await page.waitForTimeout(60);
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `宽度 ${width} 无页面横向溢出`);
  }
  await page.locator('#btn-mobile-edit').click();
  check(await input('/basic_info/name').isVisible(), '手机编辑模式');
  await input('/basic_info/name').fill('手机实时编辑');
  await settled();
  await page.locator('#btn-mobile-preview').click();
  check(await preview('/basic_info/name').first().textContent() === '手机实时编辑', '手机预览同步');
  await page.evaluate((results) => { window.__resumeEditorCheck = results; }, checks);
}
