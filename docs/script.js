document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Scroll Reveal Animation using Intersection Observer
    const revealElements = document.querySelectorAll('.section-reveal');
    
    const revealOptions = {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    };

    const revealOnScroll = new IntersectionObserver(function(entries, observer) {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                return;
            } else {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, revealOptions);

    revealElements.forEach(el => {
        revealOnScroll.observe(el);
    });

    // 2. Dynamic Number Counter for the Score
    const scoreElement = document.getElementById('final-score');
    let counted = false;

    const countUpOptions = {
        threshold: 0.5
    };

    const countUpObserver = new IntersectionObserver(function(entries, observer) {
        entries.forEach(entry => {
            if (entry.isIntersecting && !counted) {
                counted = true;
                const target = parseFloat(scoreElement.getAttribute('data-target'));
                const duration = 2000; // 2 seconds
                const frameRate = 30;
                const totalFrames = (duration / 1000) * frameRate;
                let currentFrame = 0;

                const counter = setInterval(() => {
                    currentFrame++;
                    const progress = currentFrame / totalFrames;
                    
                    // Ease out expo formula for smooth deceleration
                    const easeOutProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
                    
                    const currentScore = (target * easeOutProgress).toFixed(2);
                    scoreElement.innerText = currentScore;

                    if (currentFrame >= totalFrames) {
                        clearInterval(counter);
                        scoreElement.innerText = target.toFixed(2); // Ensure exact final value
                    }
                }, 1000 / frameRate);
                
                observer.unobserve(entry.target);
            }
        });
    }, countUpOptions);

    if (scoreElement) {
        countUpObserver.observe(scoreElement);
    }
});
