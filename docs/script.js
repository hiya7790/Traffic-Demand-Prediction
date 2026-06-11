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

    // 2. Handle Form Submission for Live Predictions
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');
    const liveScore = document.getElementById('live-score');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Get form values
            const payload = {
                geohash: document.getElementById('geohash').value,
                time: document.getElementById('time').value,
                weather: document.getElementById('weather').value,
                temperature: parseFloat(document.getElementById('temperature').value),
                population: parseFloat(document.getElementById('population').value),
                is_holiday: document.getElementById('is_holiday').checked ? 1 : 0
            };

            const submitBtn = form.querySelector('.submit-btn');
            submitBtn.innerText = "Predicting...";

            try {
                // Fetch prediction from FastAPI backend
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error('Network response was not ok');
                
                const data = await response.json();
                
                // Display result
                resultDiv.style.display = 'block';
                liveScore.innerText = data.demand.toFixed(4);
                
                // Add a little pop animation
                liveScore.style.transform = 'scale(1.2)';
                setTimeout(() => {
                    liveScore.style.transform = 'scale(1)';
                }, 200);

            } catch (error) {
                console.error("Error predicting:", error);
                alert("Error connecting to the prediction server. Make sure FastAPI is running!");
            } finally {
                submitBtn.innerText = "Predict Traffic Demand";
            }
        });
    }
});
