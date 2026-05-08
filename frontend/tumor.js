<input type="file" id="file" />
<button onclick="predict()">Predict (Tumor Detection)</button>
<button onclick="predictAlzheimer()">Predict (Alzheimer)</button>
<p id="res"></p>
<img id="heatmap" style="max-width:600px; margin-top:20px;" />

<script>
async function predict(){
    const file = document.getElementById("file").files[0];
    if(!file) {
        alert('Please select a file');
        return;
    }
    
    const fd = new FormData();
    fd.append("image", file);

    try {
        const res = await fetch("http://localhost:5000/predict/tumor", {
            method: "POST",
            body: fd
        });

        if(!res.ok) {
            const error = await res.json();
            alert('Error: ' + (error.error || 'Prediction failed'));
            return;
        }

        const data = await res.json();

        const confidence = (data.confidence * 100).toFixed(1);
        document.getElementById("res").innerText =
            data.prediction + " (Confidence: " + confidence + "%)\n" +
            "Full probabilities: " + Object.entries(data.probabilities || {})
                .map(([k, v]) => k + ": " + (v * 100).toFixed(1) + "%")
                .join(", ");

        if(data.heatmap){
            document.getElementById("heatmap").src =
                "data:image/png;base64," + data.heatmap;
        }
    } catch(error) {
        alert('Connection error: ' + error.message);
        console.error(error);
    }
}

async function predictAlzheimer(){
    const file = document.getElementById("file").files[0];
    if(!file) {
        alert('Please select a file');
        return;
    }
    
    const fd = new FormData();
    fd.append("image", file);

    try {
        const res = await fetch("http://localhost:5000/predict/alzheimer", {
            method: "POST",
            body: fd
        });

        if(!res.ok) {
            const error = await res.json();
            alert('Error: ' + (error.error || 'Prediction failed'));
            return;
        }

        const data = await res.json();

        const confidence = (data.confidence * 100).toFixed(1);
        document.getElementById("res").innerText =
            data.prediction + " (Confidence: " + confidence + "%)\n" +
            "Full probabilities: " + Object.entries(data.probabilities || {})
                .map(([k, v]) => k + ": " + (v * 100).toFixed(1) + "%")
                .join(", ");

        if(data.heatmap){
            document.getElementById("heatmap").src =
                "data:image/png;base64," + data.heatmap;
        }
    } catch(error) {
        alert('Connection error: ' + error.message);
        console.error(error);
    }
}