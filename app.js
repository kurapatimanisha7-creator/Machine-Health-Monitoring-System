document.addEventListener('DOMContentLoaded', () => {
  const cards = document.querySelectorAll('.card');
  cards.forEach((card, i) => {
    card.style.animationDelay = `${i * 80}ms`;
    card.classList.add('animate__animated', 'animate__fadeInUp');
  });
});
