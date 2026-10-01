/* Runs only in the isolated, interactive resume preview. */
(() => {
  'use strict';

  const selector = '[data-field-path]';
  const fields = Array.from(document.querySelectorAll(selector));
  const byPath = new Map();
  for (const field of fields) {
    const path = field.dataset.fieldPath;
    if (!byPath.has(path)) byPath.set(path, []);
    byPath.get(path).push(field);
  }

  const style = document.createElement('style');
  style.textContent = `
    [data-field-path] { cursor: pointer; }
    [data-field-path]:hover,
    [data-field-path].resume-field-active {
      outline: 1px dashed #8b5cf6;
      outline-offset: 2px;
      background-color: rgba(139, 92, 246, .09);
      border-radius: 2px;
      -webkit-box-decoration-break: clone;
      box-decoration-break: clone;
    }
    @media print {
      [data-field-path], [data-field-path]:hover,
      [data-field-path].resume-field-active {
        outline: none !important;
        background-color: transparent !important;
        cursor: auto;
      }
    }
  `;
  document.head.appendChild(style);

  const send = (message) => window.parent.postMessage(message, '*');
  const fieldAt = (target) => target instanceof Element ? target.closest(selector) : null;
  let hoveredPath = null;
  const hover = (path) => {
    if (path === hoveredPath) return;
    hoveredPath = path;
    send({type: 'resume-field-hover', path});
  };
  document.addEventListener('mouseover', (event) => {
    const field = fieldAt(event.target);
    hover(field ? field.dataset.fieldPath : null);
  });
  document.addEventListener('mouseout', (event) => {
    const field = fieldAt(event.relatedTarget);
    hover(field ? field.dataset.fieldPath : null);
  });
  document.addEventListener('mouseleave', () => hover(null));
  const preventLinkNavigation = (event) => {
    if (event.target instanceof Element && event.target.closest('a')) event.preventDefault();
  };
  document.addEventListener('auxclick', preventLinkNavigation);
  document.addEventListener('click', (event) => {
    preventLinkNavigation(event);
    const field = fieldAt(event.target);
    if (field) send({type: 'resume-field-select', path: field.dataset.fieldPath});
  });

  let activePath = null;
  let lastHeight = 0;
  let heightFrame = 0;
  const reportHeight = () => {
    heightFrame = 0;
    // Body has a natural A4 minimum. Root/viewport height would feed the
    // parent iframe resizing back into this measurement and prevent shrinking.
    const height = Math.ceil(Math.max(1123, document.body.getBoundingClientRect().height));
    if (height === lastHeight) return;
    lastHeight = height;
    send({type: 'resume-preview-ready', height});
  };
  const scheduleHeight = () => {
    if (!heightFrame) heightFrame = requestAnimationFrame(reportHeight);
  };

  window.addEventListener('message', (event) => {
    if (event.source !== window.parent) return;
    const message = event.data;
    if (!message || typeof message !== 'object') return;
    if (message.type === 'resume-field-active') {
      if (message.path !== null && typeof message.path !== 'string') return;
      for (const field of byPath.get(activePath) || []) field.classList.remove('resume-field-active');
      activePath = message.path;
      const matches = byPath.get(activePath) || [];
      for (const field of matches) field.classList.add('resume-field-active');
      if (message.reveal === true && matches.length) {
        const center = window.innerHeight / 2;
        const nearest = matches.reduce((a, b) =>
          Math.abs(a.getBoundingClientRect().top - center) <= Math.abs(b.getBoundingClientRect().top - center) ? a : b);
        nearest.scrollIntoView({block: 'nearest', inline: 'nearest'});
      }
    } else if (message.type === 'resume-field-value') {
      if (typeof message.path !== 'string' || typeof message.value !== 'string') return;
      for (const field of byPath.get(message.path) || []) {
        // Group titles map to their array container, never to a string leaf.
        if (field.dataset.fieldKind !== 'group-name') field.textContent = message.value;
      }
      scheduleHeight();
    }
  });

  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(scheduleHeight).observe(document.body);
  window.addEventListener('load', scheduleHeight);
  window.addEventListener('resize', scheduleHeight);
  document.fonts?.ready.then(scheduleHeight);
  scheduleHeight();
})();
