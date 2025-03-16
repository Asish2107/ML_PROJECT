document.addEventListener('DOMContentLoaded', function () {
    const resultBox = document.querySelector('.result');
    if (resultBox && resultBox.textContent.trim() !== '') {
        resultBox.style.transition = 'transform 0.5s ease, opacity 0.5s';
        resultBox.style.transform = 'scale(1.05)';
        setTimeout(() => resultBox.style.transform = 'scale(1)', 600);
    }
});