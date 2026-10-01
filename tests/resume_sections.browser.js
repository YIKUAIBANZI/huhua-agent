// Local interaction regression: playwright-cli run-code --filename=tests/resume_sections.browser.js
async (page) => {
  const checks = [];
  const check = (condition, name) => { if (!condition) throw new Error(name); checks.push(name); };
  const item = (id) => page.locator(`.edit-section[data-section-id="${id}"]`);
  const input = (path) => page.locator(`[data-input-path="${path}"]`);
  const preview = (path) => page.frameLocator('#preview-frame').locator(`[data-field-path="${path}"]`);
  const settled = () => page.waitForFunction(() => document.getElementById('render-status').textContent === '已实时更新');
  const order = () => page.locator('.section-drag-handle').evaluateAll((nodes) => nodes.map((node) => node.closest('details').dataset.sectionId));
  const headings = () => page.frameLocator('#preview-frame').locator('.section-title [data-field-path^="/sections/"]').allTextContents();
  await page.setViewportSize({width: 1440, height: 960});
  await page.goto('http://127.0.0.1:8765/resume');
  await settled();
  await page.evaluate(() => { for (const node of document.querySelectorAll('.edit-section')) if (node.dataset.sectionId !== 'basic-info') node.open = false; });

  await item('projects').locator('.section-drag-handle').dragTo(item('education').locator('summary'), {targetPosition: {x: 80, y: 2}});
  await settled();
  check((await order())[0] === 'projects', '原生拖拽调整栏目顺序');
  check((await headings())[0] === '项目经历', '预览跟随拖拽顺序');
  check(await item('projects').evaluate((node) => !node.open), '重排保留栏目折叠状态');
  await item('projects').getByRole('button', {name: '改名', exact: true}).click();
  await input('/sections/0/title').fill('科研与开源');
  await input('/projects/0/description').fill('更新项目内容');
  await settled();
  check(await item('projects').locator('.section-caption').textContent() === '科研与开源', '编辑栏目黑色标题');
  check(await preview('/sections/0/title').textContent() === '科研与开源', '预览标题实时同步');
  check(await preview('/projects/0/description').textContent() === '更新项目内容', '栏目内原内容仍可编辑');

  await item('education').getByRole('button', {name: '改名', exact: true}).click();
  await input('/sections/1/title').fill('学术经历');
  await item('education').getByRole('button', {name: '删除栏目', exact: true}).click();
  await settled();
  check(await page.frameLocator('#preview-frame').locator('[data-field-path^="/education/"]').count() === 0, '删除教育栏目不残留正文或头部别名');
  await page.getByRole('button', {name: '撤销', exact: true}).click();
  await settled();
  check(await preview('/sections/1/title').textContent() === '学术经历', '撤销保留原标题和位置');
  check(await input('/education/0/school').inputValue() === '胡话简历大学', '撤销保留原内容');
  await item('education').getByRole('button', {name: '删除栏目', exact: true}).click();
  await settled();

  await page.getByRole('button', {name: '＋ 添加自定义栏目', exact: true}).click();
  const customId = (await order()).at(-1);
  const customIndex = (await order()).length - 1;
  await input(`/sections/${customIndex}/title`).fill('竞赛获奖');
  await input(`/sections/${customIndex}/content`).fill('校级一等奖\n团队负责人 <script>window.__xss=true</script>');
  await settled();
  check(await preview(`/sections/${customIndex}/content`).textContent() === '校级一等奖\n团队负责人 <script>window.__xss=true</script>', '自定义正文支持换行与安全纯文字');
  check(await preview(`/sections/${customIndex}/content`).evaluate((node) => getComputedStyle(node.parentElement).whiteSpace) === 'pre-line', '自定义换行实际呈现');
  check(await preview(`/sections/${customIndex}/content`).evaluate(() => window.__xss !== true), '自定义内容不执行脚本');
  while ((await order())[0] !== customId) await item(customId).getByRole('button', {name: '上移栏目', exact: true}).click();
  await settled();
  check((await headings())[0] === '竞赛获奖', '上移按钮调整栏目顺序');
  check(await preview('/sections/0/content').textContent() === '校级一等奖\n团队负责人 <script>window.__xss=true</script>', '重排后自定义字段对应正确');
  await input('/sections/0/title').fill('');
  await settled();
  check(await preview('/sections/0/title').count() === 0 && await preview('/sections/0/content').count() === 1, '空标题自定义块仅显示正文');
  await input('/sections/0/title').fill('竞赛获奖');
  await settled();
  const expected = await headings();
  for (let index = 0; index < 6; index += 1) {
    await page.locator('.tpl-item').nth(index).click();
    await settled();
    check(JSON.stringify(await headings()) === JSON.stringify(expected), `模板 ${index + 1} 栏目顺序和标题一致`);
  }

  await page.locator('#btn-export').click();
  const requestPromise = page.waitForRequest((request) => request.url().includes('/api/resume/export/docx/'));
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', {name: '通用 Word', exact: true}).click();
  const request = await requestPromise;
  const payload = request.postDataJSON();
  const download = await downloadPromise;
  await download.saveAs('/tmp/huhua-sections-export.docx');
  check(payload.sections[0].id === customId && !payload.sections.some((section) => section.kind === 'education'), 'Word发送当前顺序及删除状态');
  check(payload.sections[0].content.includes('\n'), 'Word发送自定义正文');

  await page.evaluate(() => { const native = window.open; window.open = function(...args) { const popup = native.apply(this, args); popup.print = () => { popup.__printCalled = true; }; return popup; }; });
  const popupPromise = page.waitForEvent('popup');
  await page.locator('#btn-export').click();
  await page.getByRole('button', {name: '导出 PDF', exact: true}).click();
  const popup = await popupPromise;
  await popup.waitForFunction(() => window.__printCalled === true);
  check(JSON.stringify(await popup.locator('.section-title [data-field-path^="/sections/"]').allTextContents()) === JSON.stringify(expected), 'PDF打印窗口栏目顺序和标题一致');
  check(await popup.locator('[data-field-path^="/education/"]').count() === 0, 'PDF不残留删除栏目');
  check(await popup.locator('script[src*="resume-preview"]').count() === 0, 'PDF不含编辑脚本');
  await popup.close();

  await page.setViewportSize({width: 390, height: 844});
  await page.locator('#btn-mobile-edit').click();
  await item(customId).getByRole('button', {name: '下移栏目', exact: true}).click();
  await settled();
  check((await order())[1] === customId, '手机可通过按钮调整顺序');
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), '手机栏目工具无横向溢出');
  while (await page.locator('.section-tools [aria-label="删除栏目"]').count()) await page.locator('.section-tools [aria-label="删除栏目"]').first().click();
  await settled();
  check(await page.frameLocator('#preview-frame').locator('[data-field-path^="/sections/"]').count() === 0, '删除全部栏目不回退默认结构');
  await page.locator('.section-restore > summary').click();
  await page.getByRole('button', {name: '教育背景', exact: true}).click();
  await settled();
  check(await item('education').locator('.section-caption').textContent() === '学术经历', '重新添加保留已改栏目标题');
  check(await input('/education/0/school').inputValue() === '胡话简历大学', '重新添加保留原内容');
  await page.evaluate((result) => { window.__resumeSectionsCheck = result; }, checks);
}
