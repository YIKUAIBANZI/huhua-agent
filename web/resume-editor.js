(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const main = $('resume-main');
  const editor = $('form-editor');
  const frame = $('preview-frame');
  const previewScroll = $('preview-scroll');
  const previewPage = $('preview-page');
  const status = $('render-status');
  const controls = new Map();
  const pendingValues = new Map();
  const builtinSections = [
    {id: 'education', kind: 'education', title: '教育背景'},
    {id: 'work_experience', kind: 'work_experience', title: '实习经历'},
    {id: 'projects', kind: 'projects', title: '项目经历'},
    {id: 'skills', kind: 'skills', title: '技能证书'},
    {id: 'certificates', kind: 'certificates', title: '证书'},
    {id: 'self_evaluation', kind: 'self_evaluation', title: '自我评价'},
  ];
  const state = {
    data: null, templates: [], template: null, revision: 0,
    request: 0, controller: null, timer: null, ready: false,
    frameRevision: -1, structureRevision: 0, height: 1123, previewHover: null,
    editorHover: null, focused: null, exportBusy: false,
    draggedSection: null, undoSection: null,
  };
  const removedSections = new Map();
  const escapeKey = (key) => String(key).replace(/~/g, '~0').replace(/\//g, '~1');
  const pointer = (...keys) => '/' + keys.map(escapeKey).join('/');
  const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
  const element = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  };
  const button = (label, className, action) => {
    const node = element('button', className, label);
    node.type = 'button';
    node.addEventListener('click', action);
    return node;
  };
  const setStatus = (message, error = false) => {
    status.textContent = message;
    status.classList.toggle('error', error);
  };

  function normaliseData(data) {
    const copy = JSON.parse(JSON.stringify(data || {}));
    if (!copy.basic_info || typeof copy.basic_info !== 'object') copy.basic_info = {};
    for (const key of ['education', 'work_experience', 'projects', 'skills', 'certificates']) {
      if (!Array.isArray(copy[key])) copy[key] = [];
    }
    if (!copy.skill_groups || Array.isArray(copy.skill_groups) || typeof copy.skill_groups !== 'object') copy.skill_groups = {};
    for (const key of Object.keys(copy.skill_groups)) {
      if (!Array.isArray(copy.skill_groups[key])) delete copy.skill_groups[key];
    }
    for (const item of copy.work_experience) if (!Array.isArray(item.bullets)) item.bullets = [];
    for (const item of copy.projects) if (!Array.isArray(item.polished_bullets)) item.polished_bullets = [];
    if (!Array.isArray(copy.sections)) {
      copy.sections = builtinSections.map((section) => ({...section, content: '', title: section.kind === 'projects' ? copy.experience_section_title || section.title : section.title}));
    }
    return copy;
  }

  function setValue(path, value) {
    const keys = path.slice(1).split('/').map((key) => key.replace(/~1/g, '/').replace(/~0/g, '~'));
    let parent = state.data;
    for (const key of keys.slice(0, -1)) {
      if (!parent || !own(parent, key)) return;
      parent = parent[key];
    }
    if (!parent || typeof parent !== 'object') return;
    Object.defineProperty(parent, keys[keys.length - 1], {value, writable: true, enumerable: true, configurable: true});
  }

  function post(message) {
    if (state.ready) frame.contentWindow.postMessage(message, '*');
  }

  function changed(path, value) {
    state.revision += 1;
    state.controller?.abort();
    setStatus('正在更新');
    if (path) {
      pendingValues.set(path, {revision: state.revision, value});
      post({type: 'resume-field-value', path, value});
    }
    clearTimeout(state.timer);
    state.timer = setTimeout(renderPreview, 180);
  }

  function changeStructure(action) {
    post({type: 'resume-field-active', path: null, reveal: false});
    state.ready = false;
    state.structureRevision = state.revision + 1;
    action();
    pendingValues.clear();
    state.previewHover = null;
    state.editorHover = null;
    state.focused = null;
    buildForm();
    changed();
  }

  function field(container, label, path, value, options = {}) {
    const wrapper = element('div', 'field' + (options.full ? ' full' : ''));
    wrapper.dataset.fieldPath = path;
    const name = element('label', '', label);
    const input = element(options.multiline ? 'textarea' : 'input');
    input.id = 'resume-field-' + controls.size;
    input.dataset.inputPath = path;
    input.value = typeof value === 'string' ? value : '';
    if (!options.multiline) input.type = 'text';
    else input.rows = options.rows || 3;
    if (options.placeholder) input.placeholder = options.placeholder;
    if (options.autocomplete) input.autocomplete = options.autocomplete;
    if (options.maxLength) input.maxLength = options.maxLength;
    name.htmlFor = input.id;
    wrapper.append(name, input);
    container.append(wrapper);
    controls.set(path, {wrapper, input});
    input.addEventListener('input', () => {
      setValue(input.dataset.inputPath, input.value);
      options.onInput?.(input.value);
      changed(input.dataset.inputPath, input.value);
    });
    return wrapper;
  }

  function section(title, meta, index, openStates) {
    const details = element('details', 'edit-section');
    const id = meta?.id || 'basic-info';
    details.dataset.sectionId = id;
    details.open = openStates?.get(id) ?? true;
    const summary = element('summary', 'section-summary');
    const caption = element('span', 'section-caption', title || '未命名栏目');
    if (meta) {
      const handle = button('⠿', 'section-drag-handle', () => {});
      handle.draggable = true;
      handle.setAttribute('aria-label', '拖拽调整栏目顺序');
      handle.title = '拖拽调整栏目顺序';
      summary.append(handle);
      const actions = element('span', 'section-tools');
      const rename = button('改名', 'section-tool', () => {
        details.open = true;
        revealControl(pointer('sections', index, 'title'), true);
        controls.get(pointer('sections', index, 'title')).input.select();
      });
      const up = button('↑', 'section-tool', () => moveSection(id, -1));
      const down = button('↓', 'section-tool', () => moveSection(id, 1));
      up.setAttribute('aria-label', '上移栏目');
      down.setAttribute('aria-label', '下移栏目');
      up.disabled = index === 0;
      down.disabled = index === state.data.sections.length - 1;
      const remove = button('×', 'section-tool section-remove', () => {
        changeStructure(() => {
          state.undoSection = {section: {...meta}, index};
          if (meta.kind !== 'custom') removedSections.set(meta.kind, {...meta});
          state.data.sections.splice(index, 1);
        });
      });
      remove.setAttribute('aria-label', '删除栏目');
      remove.title = '删除此栏目';
      actions.append(rename, up, down, remove);
      summary.append(caption, actions);
      for (const control of [handle, rename, up, down, remove]) {
        control.addEventListener('click', (event) => { event.preventDefault(); event.stopPropagation(); });
      }
      handle.addEventListener('dragstart', (event) => {
        state.draggedSection = id;
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', id);
        details.classList.add('section-dragging');
      });
      handle.addEventListener('dragend', clearSectionDrag);
      details.addEventListener('dragover', (event) => {
        if (!state.draggedSection || state.draggedSection === id) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
        const below = event.clientY > summary.getBoundingClientRect().top + summary.offsetHeight / 2;
        for (const node of editor.querySelectorAll('.drop-before, .drop-after')) node.classList.remove('drop-before', 'drop-after');
        details.classList.add(below ? 'drop-after' : 'drop-before');
      });
      details.addEventListener('dragleave', (event) => {
        if (!details.contains(event.relatedTarget)) details.classList.remove('drop-before', 'drop-after');
      });
      details.addEventListener('drop', (event) => {
        if (!state.draggedSection) return;
        event.preventDefault();
        const draggedId = state.draggedSection;
        const below = details.classList.contains('drop-after');
        clearSectionDrag();
        if (draggedId === id) return;
        changeStructure(() => {
          const sections = state.data.sections;
          const source = sections.findIndex((item) => item.id === draggedId);
          if (source < 0) return;
          const [moving] = sections.splice(source, 1);
          const target = sections.findIndex((item) => item.id === id);
          sections.splice(target + (below ? 1 : 0), 0, moving);
        });
        focusSection(draggedId);
      });
    } else summary.append(caption);
    details.append(summary);
    const body = element('div', 'section-body');
    details.append(body);
    editor.append(details);
    if (meta) {
      field(body, '栏目标题', pointer('sections', index, 'title'), meta.title, {
        maxLength: 100, placeholder: meta.kind === 'custom' ? '留空只显示正文内容' : '填写栏目名称',
        onInput: (value) => { caption.textContent = value || '未命名栏目'; },
      });
    }
    return body;
  }

  function clearSectionDrag() {
    state.draggedSection = null;
    for (const node of editor.querySelectorAll('.section-dragging, .drop-before, .drop-after')) node.classList.remove('section-dragging', 'drop-before', 'drop-after');
  }

  function focusSection(id, editTitle = false) {
    const index = state.data.sections.findIndex((item) => item.id === id);
    if (index < 0) return;
    const details = [...editor.querySelectorAll('.edit-section')].find((node) => node.dataset.sectionId === id);
    if (editTitle) {
      details.open = true;
      revealControl(pointer('sections', index, 'title'), true);
      controls.get(pointer('sections', index, 'title')).input.select();
    } else {
      details.querySelector('.section-drag-handle').focus({preventScroll: true});
      const box = details.querySelector('summary').getBoundingClientRect();
      const viewport = editor.getBoundingClientRect();
      if (box.top < viewport.top || box.bottom > viewport.bottom) editor.scrollTop += box.top - viewport.top - 20;
    }
  }

  function moveSection(id, offset) {
    const index = state.data.sections.findIndex((item) => item.id === id);
    const target = index + offset;
    if (index < 0 || target < 0 || target >= state.data.sections.length) return;
    changeStructure(() => {
      const [moving] = state.data.sections.splice(index, 1);
      state.data.sections.splice(target, 0, moving);
    });
    focusSection(id);
  }

  function buildSectionActions() {
    const footer = element('div', 'section-actions');
    footer.append(element('p', 'section-instructions', '拖动 ⠿ 调整栏目顺序，也可用 ↑ ↓ 移动。'));
    if (state.undoSection) {
      const notice = element('div', 'section-undo');
      notice.setAttribute('role', 'status');
      notice.append(element('span', '', `已删除「${state.undoSection.section.title || '未命名栏目'}」`));
      notice.append(button('撤销', 'text-button', () => {
        if (state.data.sections.length >= 30) { setStatus('最多添加 30 个栏目，请先移除一个', true); return; }
        const removed = state.undoSection;
        if (state.data.sections.some((item) => item.id === removed.section.id || (item.kind !== 'custom' && item.kind === removed.section.kind))) {
          state.undoSection = null;
          buildForm();
          return;
        }
        changeStructure(() => {
          state.data.sections.splice(Math.min(removed.index, state.data.sections.length), 0, removed.section);
          state.undoSection = null;
        });
        focusSection(removed.section.id);
      }));
      footer.append(notice);
    }
    const custom = button('＋ 添加自定义栏目', 'add-button', () => {
      const id = 'custom-' + crypto.randomUUID();
      changeStructure(() => state.data.sections.push({id, kind: 'custom', title: '自定义栏目', content: ''}));
      focusSection(id, true);
    });
    custom.disabled = state.data.sections.length >= 30;
    footer.append(custom);
    const missing = builtinSections.filter((builtin) => !state.data.sections.some((item) => item.kind === builtin.kind));
    if (missing.length) {
      const restore = element('details', 'section-restore');
      restore.append(element('summary', '', '添加已有栏目'));
      const options = element('div', 'section-restore-options');
      for (const builtin of missing) {
        const option = button(builtin.title, 'secondary-button', () => {
          changeStructure(() => {
            const restored = removedSections.get(builtin.kind) || {...builtin, content: '', title: builtin.kind === 'projects' ? state.data.experience_section_title || builtin.title : builtin.title};
            state.data.sections.push({...restored});
            if (state.undoSection?.section.kind === builtin.kind) state.undoSection = null;
          });
          focusSection(builtin.id, true);
        });
        option.disabled = state.data.sections.length >= 30;
        options.append(option);
      }
      restore.append(options);
      footer.append(restore);
    }
    editor.append(footer);
  }

  function itemEditor(container, title, removeAction) {
    const item = element('div', 'item-editor');
    const heading = element('div', 'item-heading');
    heading.append(element('span', 'item-title', title), button('删除', 'text-button remove', () => changeStructure(removeAction)));
    item.append(heading);
    container.append(item);
    return item;
  }

  function stringList(container, values, keys, label, multiline = false, afterRemove = () => {}) {
    const list = element('div', 'list-editor');
    values.forEach((value, index) => {
      const row = element('div', 'list-field-row');
      field(row, `${label} ${index + 1}`, pointer(...keys, index), value, {multiline, rows: 3});
      const remove = button('×', 'remove-line', () => changeStructure(() => { values.splice(index, 1); afterRemove(); }));
      remove.setAttribute('aria-label', `删除${label} ${index + 1}`);
      row.append(remove);
      list.append(row);
    });
    list.append(button(`＋ 添加${label}`, 'add-button', () => changeStructure(() => values.push(''))));
    container.append(list);
  }

  function buildForm() {
    const scrollTop = editor.scrollTop;
    const openStates = new Map([...editor.querySelectorAll('.edit-section')].map((node) => [node.dataset.sectionId, node.open]));
    controls.clear();
    editor.replaceChildren();
    const basic = section('基本信息', null, -1, openStates);
    buildBasicContent(basic);
    const builders = {
      education: buildEducationContent, work_experience: buildWorkContent,
      projects: buildProjectsContent, skills: buildSkillsContent,
      certificates: buildCertificatesContent, self_evaluation: buildEvaluationContent,
    };
    state.data.sections.forEach((meta, index) => {
      const body = section(meta.title, meta, index, openStates);
      if (meta.kind === 'custom') {
        field(body, '正文内容', pointer('sections', index, 'content'), meta.content, {
          multiline: true, rows: 6, maxLength: 20000,
          placeholder: '写下竞赛获奖、校园活动、开源贡献或其他想展示的内容，可换行。',
        });
      } else builders[meta.kind]?.(body);
    });
    buildSectionActions();
    editor.scrollTop = scrollTop;
  }

  function buildBasicContent(basic) {
    const data = state.data;
    const basicGrid = element('div', 'field-grid');
    basic.append(basicGrid);
    for (const [key, label, full, autocomplete] of [
      ['name', '姓名', false, 'name'], ['phone', '电话', false, 'tel'],
      ['email', '邮箱', true, 'email'], ['location', '所在城市'],
      ['hometown', '籍贯'], ['objective', '求职意向', true],
      ['gender', '性别'], ['birthday', '出生年月'], ['age', '年龄'],
      ['ethnicity', '民族'], ['political', '政治面貌'], ['height', '身高'],
    ]) field(basicGrid, label, pointer('basic_info', key), data.basic_info[key], {full, autocomplete});
    if (data.basic_info.photo) {
      const actions = element('div', 'photo-actions');
      actions.append(button('移除头像', 'text-button remove', () => changeStructure(() => { data.basic_info.photo = ''; })));
      basic.append(actions);
    }
  }

  function buildEducationContent(education) {
    const data = state.data;
    data.education.forEach((item, index) => {
      const box = itemEditor(education, `教育经历 ${index + 1}`, () => data.education.splice(index, 1));
      const grid = element('div', 'field-grid');
      box.append(grid);
      for (const [key, label, full] of [['school', '学校', true], ['major', '专业'], ['degree', '学历'], ['start_date', '开始时间'], ['end_date', '结束时间']]) {
        field(grid, label, pointer('education', index, key), item[key], {full});
      }
      field(grid, '课程与成绩', pointer('education', index, 'highlights'), item.highlights, {full: true, multiline: true});
    });
    education.append(button('＋ 添加教育经历', 'add-button', () => changeStructure(() => data.education.push({school: '', major: '', degree: '', start_date: '', end_date: '', highlights: ''}))));
  }

  function buildWorkContent(work) {
    const data = state.data;
    data.work_experience.forEach((item, index) => {
      const box = itemEditor(work, `经历 ${index + 1}`, () => data.work_experience.splice(index, 1));
      const grid = element('div', 'field-grid');
      box.append(grid);
      for (const [key, label, full] of [['company', '公司 / 组织', true], ['title', '岗位', true], ['start_date', '开始时间'], ['end_date', '结束时间']]) {
        field(grid, label, pointer('work_experience', index, key), item[key], {full});
      }
      stringList(box, item.bullets, ['work_experience', index, 'bullets'], '经历要点', true);
    });
    work.append(button('＋ 添加工作 / 实习经历', 'add-button', () => changeStructure(() => data.work_experience.push({company: '', title: '', start_date: '', end_date: '', bullets: ['']}))));
  }

  function buildProjectsContent(projects) {
    const data = state.data;
    data.projects.forEach((item, index) => {
      const box = itemEditor(projects, `项目 ${index + 1}`, () => data.projects.splice(index, 1));
      const grid = element('div', 'field-grid');
      box.append(grid);
      for (const [key, label, full] of [['name', '项目名称', true], ['role', '担任角色', true], ['start_date', '开始时间'], ['end_date', '结束时间']]) {
        field(grid, label, pointer('projects', index, key), item[key], {full});
      }
      if (item.polished_bullets.length) {
        stringList(box, item.polished_bullets, ['projects', index, 'polished_bullets'], '项目要点', true, () => {
          if (!item.polished_bullets.length) item.description = '';
        });
        box.append(button('改为整段描述', 'text-button', () => changeStructure(() => {
          item.description = item.polished_bullets.join('\n');
          item.polished_bullets = [];
        })));
      } else {
        field(grid, '项目描述', pointer('projects', index, 'description'), item.description, {full: true, multiline: true, rows: 4});
        box.append(button('改为分条要点', 'text-button', () => changeStructure(() => {
          item.polished_bullets = item.description ? [item.description] : [''];
          item.description = '';
        })));
      }
    });
    projects.append(button('＋ 添加项目', 'add-button', () => changeStructure(() => data.projects.push({name: '', role: '', start_date: '', end_date: '', description: '', polished_bullets: []}))));
  }

  function buildSkillsContent(skills) {
    const data = state.data;
    const groups = Object.entries(data.skill_groups);
    if (groups.length) {
      for (const [initialName, values] of groups) {
        let name = initialName;
        const group = element('div', 'group-editor');
        let path = pointer('skill_groups', name);
        group.dataset.fieldPath = path;
        const heading = element('div', 'item-heading');
        heading.append(element('span', 'item-title', '技能分组'), button('删除分组', 'text-button remove', () => changeStructure(() => {
          delete data.skill_groups[name];
          if (!Object.keys(data.skill_groups).length) data.skills = [];
        })));
        group.append(heading);
        const nameField = element('div', 'field');
        nameField.dataset.fieldPath = path;
        const nameLabel = element('label', '', '分组名称');
        const nameInput = element('input');
        nameInput.type = 'text';
        nameInput.id = 'resume-field-' + controls.size;
        nameInput.value = name;
        nameInput.setAttribute('aria-label', '分组名称');
        nameLabel.htmlFor = nameInput.id;
        nameField.append(nameLabel, nameInput);
        group.append(nameField);
        controls.set(path, {wrapper: nameField, input: nameInput});
        nameInput.addEventListener('input', () => {
          const next = nameInput.value;
          if (!next.trim() || (next !== name && own(data.skill_groups, next))) {
            nameInput.setCustomValidity(!next.trim() ? '请填写分组名称' : '分组名称已存在');
            return;
          }
          nameInput.setCustomValidity('');
          if (next === name) return;
          const newGroups = {};
          for (const [key, items] of Object.entries(data.skill_groups)) {
            Object.defineProperty(newGroups, key === name ? next : key, {value: items, writable: true, enumerable: true, configurable: true});
          }
          data.skill_groups = newGroups;
          const nextPath = pointer('skill_groups', next);
          for (const [oldPath, control] of [...controls]) {
            if (oldPath !== path && !oldPath.startsWith(path + '/')) continue;
            const newPath = nextPath + oldPath.slice(path.length);
            controls.delete(oldPath);
            controls.set(newPath, control);
            control.wrapper.dataset.fieldPath = newPath;
            if (control.input.dataset.inputPath) control.input.dataset.inputPath = newPath;
          }
          group.dataset.fieldPath = nextPath;
          state.focused = nextPath;
          if (state.editorHover === path || state.editorHover?.startsWith(path + '/')) state.editorHover = nextPath + state.editorHover.slice(path.length);
          name = next;
          path = nextPath;
          pendingValues.clear();
          state.ready = false;
          state.structureRevision = state.revision + 1;
          changed();
        });
        nameInput.addEventListener('blur', () => { if (!nameInput.checkValidity()) { nameInput.value = name; nameInput.setCustomValidity(''); } });
        stringList(group, values, ['skill_groups', name], '技能');
        skills.append(group);
      }
      skills.append(button('改为不分组技能', 'text-button', () => changeStructure(() => {
        data.skills = Object.entries(data.skill_groups).flatMap(([name, values]) => values.map((value) => `${name}：${value}`));
        data.skill_groups = {};
      })));
    } else {
      stringList(skills, data.skills, ['skills'], '技能');
    }
    skills.append(button('＋ 添加技能分组', 'add-button', () => {
      let name = '新分组';
      let index = 2;
      while (own(data.skill_groups, name)) name = `新分组 ${index++}`;
      changeStructure(() => {
        if (!Object.keys(data.skill_groups).length && data.skills.length) {
          Object.defineProperty(data.skill_groups, '技能', {value: [...data.skills], writable: true, enumerable: true, configurable: true});
        }
        data.skills = [];
        Object.defineProperty(data.skill_groups, name, {value: [''], writable: true, enumerable: true, configurable: true});
      });
    }));
  }

  function buildCertificatesContent(certificates) {
    const data = state.data;
    stringList(certificates, data.certificates, ['certificates'], '证书');
  }

  function buildEvaluationContent(evaluation) {
    const data = state.data;
    field(evaluation, '自我评价', pointer('self_evaluation'), data.self_evaluation, {multiline: true, rows: 4});
  }

  function revealControl(path, focus = false) {
    const control = controls.get(path);
    if (!control) return;
    for (let node = control.wrapper.parentElement; node && node !== editor; node = node.parentElement) {
      if (node.tagName === 'DETAILS') node.open = true;
    }
    if (focus) {
      main.classList.add('show-data');
      $('btn-toggle-data').textContent = '收起编辑';
      $('btn-toggle-data').setAttribute('aria-expanded', 'true');
      if (matchMedia('(max-width: 700px)').matches) mobileMode(true);
      control.input.focus({preventScroll: true});
    }
    const box = control.wrapper.getBoundingClientRect();
    const viewport = editor.getBoundingClientRect();
    if (box.top < viewport.top + 8 || box.bottom > viewport.bottom - 8) {
      editor.scrollTop += box.top - viewport.top - Math.min(60, viewport.height / 4);
    }
  }

  function highlightControls() {
    for (const [path, control] of controls) {
      control.wrapper.classList.toggle('linked', path === state.previewHover);
      control.wrapper.classList.toggle('focused', path === state.focused);
    }
  }

  function highlightPreview(reveal = false) {
    post({type: 'resume-field-active', path: state.editorHover || state.focused, reveal});
  }

  editor.addEventListener('mouseover', (event) => {
    const wrapper = event.target.closest('[data-field-path]');
    const path = wrapper?.dataset.fieldPath || null;
    if (path !== state.editorHover) {
      state.editorHover = path;
      highlightPreview();
    }
  });
  editor.addEventListener('mouseleave', () => { state.editorHover = null; highlightPreview(); });
  editor.addEventListener('focusin', (event) => {
    state.focused = event.target.closest('[data-field-path]')?.dataset.fieldPath || null;
    highlightControls();
    highlightPreview(true);
  });
  editor.addEventListener('focusout', () => {
    requestAnimationFrame(() => {
      state.focused = document.activeElement.closest?.('[data-field-path]')?.dataset.fieldPath || null;
      highlightControls();
      highlightPreview();
    });
  });

  window.addEventListener('message', (event) => {
    if (event.source !== frame.contentWindow || !event.data || typeof event.data !== 'object') return;
    const message = event.data;
    if (message.type === 'resume-preview-ready') {
      if (state.frameRevision < state.structureRevision) return;
      if (!Number.isFinite(message.height) || message.height < 1123 || message.height > 100000) return;
      state.ready = true;
      state.height = message.height;
      scalePreview();
      // Keystrokes can arrive while a new iframe document is loading.
      for (const [path, change] of pendingValues) {
        if (change.revision > state.frameRevision && controls.has(path)) {
          post({type: 'resume-field-value', path, value: change.value});
        }
      }
      highlightPreview();
      if (state.frameRevision === state.revision) setStatus('已实时更新');
    } else if (message.type === 'resume-field-hover') {
      if (!state.ready) return;
      if (message.path !== null && !controls.has(message.path)) return;
      state.previewHover = message.path;
      highlightControls();
      if (message.path) revealControl(message.path);
    } else if (message.type === 'resume-field-select' && state.ready && controls.has(message.path)) {
      revealControl(message.path, true);
    }
  });

  function scalePreview() {
    if (!previewScroll.clientWidth) return;
    const style = getComputedStyle(previewScroll);
    const width = previewScroll.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const scale = Math.min(1, Math.max(.1, width / 794));
    previewPage.style.width = `${794 * scale}px`;
    previewPage.style.height = `${state.height * scale}px`;
    frame.style.height = `${state.height}px`;
    frame.style.transform = `scale(${scale})`;
  }
  new ResizeObserver(scalePreview).observe(previewScroll);
  window.addEventListener('resize', scalePreview);

  async function responseError(response, fallback) {
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') return body.detail;
    } catch (_) { /* Non-JSON error responses use a plain, local message. */ }
    return fallback;
  }

  async function renderPreview() {
    if (!state.template || !state.data) return;
    clearTimeout(state.timer);
    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    const revision = state.revision;
    const template = state.template.id;
    const request = ++state.request;
    setStatus('正在更新');
    try {
      const response = await fetch(`/api/resume/render/${encodeURIComponent(template)}?interactive=true`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(state.data), signal: controller.signal,
      });
      if (!response.ok) throw new Error(await responseError(response, '预览更新失败，请稍后重试'));
      const html = await response.text();
      if (revision !== state.revision || request !== state.request || template !== state.template.id) return;
      state.ready = false;
      state.previewHover = null;
      highlightControls();
      state.frameRevision = revision;
      frame.srcdoc = html;
    } catch (error) {
      if (error.name === 'AbortError' || revision !== state.revision || request !== state.request) return;
      setStatus('更新失败，继续编辑可重试', true);
      status.title = error.message;
    }
  }

  function selectTemplate(template) {
    state.template = template;
    $('current-tpl').textContent = template.name;
    for (const item of $('template-list').children) {
      const active = item.dataset.templateId === template.id;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    }
    renderPreview();
  }

  function mobileMode(edit) {
    main.classList.toggle('mobile-edit', edit);
    $('btn-mobile-preview').classList.toggle('active', !edit);
    $('btn-mobile-edit').classList.toggle('active', edit);
    $('btn-mobile-preview').setAttribute('aria-pressed', String(!edit));
    $('btn-mobile-edit').setAttribute('aria-pressed', String(edit));
    requestAnimationFrame(scalePreview);
  }
  $('btn-mobile-preview').addEventListener('click', () => mobileMode(false));
  $('btn-mobile-edit').addEventListener('click', () => mobileMode(true));
  $('btn-toggle-data').addEventListener('click', () => {
    const visible = main.classList.toggle('show-data');
    $('btn-toggle-data').textContent = visible ? '收起编辑' : '编辑内容';
    $('btn-toggle-data').setAttribute('aria-expanded', String(visible));
  });

  $('btn-upload-photo').addEventListener('click', () => $('photo-input').click());
  $('photo-input').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    event.target.value = '';
    if (!file || !state.data) return;
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setStatus('头像仅支持 JPG / PNG', true);
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setStatus('头像不能超过 5 MB', true);
      return;
    }
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      changeStructure(() => { state.data.basic_info.photo = dataUrl; });
    } catch (_) { setStatus('头像读取失败，请重新选择', true); }
  });

  const exportMenu = $('export-menu');
  function closeExportMenu() {
    exportMenu.hidden = true;
    $('btn-export').setAttribute('aria-expanded', 'false');
  }
  $('btn-export').addEventListener('click', () => {
    exportMenu.hidden = !exportMenu.hidden;
    $('btn-export').setAttribute('aria-expanded', String(!exportMenu.hidden));
  });
  document.addEventListener('click', (event) => { if (!event.target.closest('#export-wrap')) closeExportMenu(); });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeExportMenu(); });

  async function exportResume(format) {
    closeExportMenu();
    if (!state.data || !state.template || state.exportBusy) return;
    const snapshot = JSON.stringify(state.data);
    const template = encodeURIComponent(state.template.id);
    // Open synchronously while the user's click still permits a print window.
    const printWindow = format === 'pdf' ? window.open('', '_blank') : null;
    if (format === 'pdf' && !printWindow) { setStatus('请允许弹出窗口后再导出 PDF', true); return; }
    if (printWindow) { printWindow.opener = null; printWindow.document.title = '准备打印简历'; }
    state.exportBusy = true;
    $('btn-export').disabled = true;
    try {
      const response = await fetch(format === 'pdf' ? `/api/resume/render/${template}` : `/api/resume/export/docx/${template}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: snapshot,
      });
      if (!response.ok) throw new Error(await responseError(response, '导出失败，请稍后重试'));
      if (format === 'pdf') {
        const html = await response.text();
        const printable = html.replace('</body>', '<script>window.addEventListener("load",()=>{document.fonts.ready.then(()=>window.print());});<\/script></body>');
        printWindow.document.open();
        printWindow.document.write(printable);
        printWindow.document.close();
      } else {
        const url = URL.createObjectURL(await response.blob());
        const link = element('a');
        link.href = url;
        let filename = '简历-通用Word.docx';
        const match = (response.headers.get('Content-Disposition') || '').match(/filename\*=UTF-8''([^;]+)/i);
        if (match) { try { filename = decodeURIComponent(match[1]); } catch (_) { /* Use default filename. */ } }
        link.download = filename;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (error) {
      printWindow?.close();
      setStatus(error.message.includes('fetch') ? '连接已断开，请确认服务运行后重试' : error.message, true);
    } finally {
      state.exportBusy = false;
      $('btn-export').disabled = false;
    }
  }
  for (const option of document.querySelectorAll('.export-opt')) option.addEventListener('click', () => exportResume(option.dataset.format));

  async function initialise() {
    try {
      let session = new URLSearchParams(location.search).get('session');
      if (!session) { try { session = localStorage.getItem('huhua_session_id'); } catch (_) { /* Private browsing may block storage. */ } }
      const templateResponse = await fetch('/api/resume/templates');
      if (!templateResponse.ok) throw new Error('模板加载失败');
      state.templates = (await templateResponse.json()).templates;
      let source = '样例';
      if (session) {
        const currentResponse = await fetch(`/api/resume/current?session_id=${encodeURIComponent(session)}`);
        if (!currentResponse.ok) throw new Error(await responseError(currentResponse, '当前简历读取失败，请刷新重试'));
        const current = await currentResponse.json();
        if (current.has_data && current.data) { state.data = current.data; source = '当前会话'; }
      }
      if (!state.data) {
        const sampleResponse = await fetch('/api/resume/sample');
        if (!sampleResponse.ok) throw new Error('样例加载失败');
        state.data = await sampleResponse.json();
      }
      state.data = normaliseData(state.data);
      $('data-source').textContent = source;
      $('tpl-count').textContent = `(${state.templates.length})`;
      for (const template of state.templates) {
        const item = button('', 'tpl-item', () => selectTemplate(template));
        item.dataset.templateId = template.id;
        item.append(element('span', 'tpl-name', template.name), element('span', 'tpl-style', template.style || ''));
        $('template-list').append(item);
      }
      buildForm();
      if (state.templates.length) selectTemplate(state.templates[0]);
      else setStatus('暂无可用模板', true);
    } catch (error) {
      setStatus(error.message.includes('fetch') ? '连接已断开，请确认服务运行后刷新' : error.message, true);
      editor.replaceChildren(element('p', 'empty-section', '暂时无法加载简历内容，请刷新页面重试。'));
    }
  }
  initialise();
})();
