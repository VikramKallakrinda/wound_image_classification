document.addEventListener('DOMContentLoaded', function () {
    const startButton = document.getElementById('start-button');
    const options = document.getElementById('options');
    const uploadSection = document.getElementById('upload-section');
    const estimatedTime = document.getElementById('estimated-time');
    const resultModal = document.getElementById('result-modal');
    const closeModal = document.querySelector('.close');

    startButton.addEventListener('click', function () {
        options.classList.remove('hidden');
    });

    document.getElementById('single-image-button').addEventListener('click', function () {
        uploadSection.classList.remove('hidden');
        estimatedTime.classList.add('hidden');
    });

    document.getElementById('multi-image-button').addEventListener('click', function () {
        uploadSection.classList.remove('hidden');
        estimatedTime.classList.remove('hidden');
        // Start countdown timer here if needed
    });

    document.getElementById('submit-button').addEventListener('click', function () {
        // Trigger file upload processing
        // Show result modal after processing
        resultModal.classList.remove('hidden');
    });

    closeModal.addEventListener('click', function () {
        resultModal.classList.add('hidden');
    });
});
