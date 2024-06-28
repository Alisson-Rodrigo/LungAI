let session;

async function loadModel() {
    try {
        // Ensure the path is correct
        session = await ort.InferenceSession.create('/lung_cancer_detection_model.onnx');
        console.log('Model loaded successfully:', session);
    } catch (error) {
        console.error('Failed to load the model:', error);
    }
}

async function predict() {
    if (!session) {
        console.error('Model session is not initialized');
        alert('Model not loaded yet. Please wait.');
        return;
    }

    const fileInput = document.getElementById('file-input');
    if (fileInput.files.length === 0) {
        alert('Please select an image.');
        return;
    }

    const image = fileInput.files[0];
    const imageData = await preprocessImage(image);

    const inputTensor = new ort.Tensor('float32', imageData, [1, 3, 250, 250]);
    try {
        const results = await session.run({ input: inputTensor });
        console.log('Model run results:', results);
        const outputTensor = results.output;
        const probabilities = softmax(outputTensor.data);
        const predictedClass = argMax(probabilities);

        // Assuming the classes are 'No Cancer' and 'Cancer'
        const classes = ['No Cancer', 'Cancer'];
        const resultText = `Prediction: ${classes[predictedClass]} (Confidence: ${probabilities[predictedClass].toFixed(4)})`;
        document.getElementById('prediction-result').textContent = resultText;
    } catch (error) {
        console.error('Failed to run the model:', error);
    }
}

function softmax(arr) {
    const exp = arr.map(Math.exp);
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(e => e / sum);
}

function argMax(arr) {
    return arr.indexOf(Math.max(...arr));
}

async function preprocessImage(image) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.src = reader.result;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 250;
                canvas.height = 250;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 250, 250);

                // Get the image data from the canvas
                const imageData = ctx.getImageData(0, 0, 250, 250);
                const { data } = imageData;

                // Preprocess the image data to match the model input
                const floatData = new Float32Array(3 * 250 * 250);
                for (let i = 0; i < 250 * 250; i++) {
                    floatData[i] = (data[i * 4] / 255 - 0.485) / 0.229;
                    floatData[i + 250 * 250] = (data[i * 4 + 1] / 255 - 0.456) / 0.224;
                    floatData[i + 2 * 250 * 250] = (data[i * 4 + 2] / 255 - 0.406) / 0.225;
                }
                resolve(floatData);
            };
            img.onerror = (error) => reject(error);
        };
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(image);
    });
}

function displayImage(image) {
    const reader = new FileReader();
    reader.onload = () => {
        const imgElement = document.getElementById('uploaded-image');
        imgElement.src = reader.result;
        imgElement.style.display = 'block'; // show the image
        console.log('Image displayed:', imgElement.src); // debug log
    };
    reader.onerror = (error) => console.error('Error reading file:', error); // debug log
    reader.readAsDataURL(image);
}

// Add event listener to display image when file is selected
document.getElementById('file-input').addEventListener('change', function() {
    if (this.files && this.files[0]) {
        displayImage(this.files[0]);
    }
});

// Load the model when the page loads
window.onload = loadModel;
