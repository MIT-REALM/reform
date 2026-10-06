'use strict';

// Every figure remains a normal image link when JavaScript is unavailable.
const figureDialog = document.querySelector('#figure-dialog');
const dialogImage = document.querySelector('#dialog-image');
const dialogTitle = document.querySelector('#figure-title');
if (figureDialog && typeof figureDialog.showModal === 'function') {
  document.querySelectorAll('[data-zoom]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const source = link.querySelector('img');
      dialogImage.src = link.href;
      dialogImage.alt = source.alt;
      dialogTitle.textContent = link.getAttribute('aria-label');
      figureDialog.showModal();
      figureDialog.scrollTop = 0;
      figureDialog.scrollLeft = 0;
      document.body.classList.add('dialog-open');
    });
  });
  document.querySelector('#close-figure').addEventListener('click', () => figureDialog.close());
  figureDialog.addEventListener('click', event => {
    if (event.target !== figureDialog) return;
    const bounds = figureDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) figureDialog.close();
  });
  figureDialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));
}

const copyButton = document.querySelector('#copy-bibtex');
const citation = document.querySelector('#citation');
const copyStatus = document.querySelector('#copy-status');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = 'Copied!';
    copyStatus.textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2500);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C (or ⌘C on Mac) to copy.';
  }
});

const researchMenu = document.querySelector('.research-menu');
document.addEventListener('click', event => {
  if (!researchMenu.contains(event.target)) researchMenu.open = false;
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && researchMenu.open) {
    researchMenu.open = false;
    researchMenu.querySelector('summary').focus();
  }
});

// Match the earlier project pages' muted, looping demos while respecting reduced motion.
// With JavaScript disabled, native controls and poster images still work.
const demoVideos = document.querySelectorAll('.demo-gallery video');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
function updateDemoPlayback() {
  demoVideos.forEach(video => {
    video.autoplay = !reducedMotion.matches;
    if (reducedMotion.matches) {
      video.pause();
    } else {
      video.muted = true;
      // Autoplay may be blocked by the browser; native controls remain available.
      video.play().catch(() => {});
    }
  });
}
updateDemoPlayback();
reducedMotion.addEventListener('change', updateDemoPlayback);
