async function generateImage() {
    const prompt = document.getElementById("image_prompt").value.trim();
    const loadingDiv = document.getElementById("loading");
    const resultImage = document.getElementById("generated-image");

    if (prompt === "") {
        alert("Please enter a prompt.");
        return;
    }

    // Show loading message
    loadingDiv.style.display = "block";
    resultImage.style.display = "none";

    try {
        const response = await fetch("/generate_image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const result = await response.json();
        resultImage.src = result.image_url;
        resultImage.style.display = "block";

    } catch (error) {
        console.error("Error generating image:", error);
        alert("Failed to generate image. Please try again.");
    } finally {
        loadingDiv.style.display = "none";
    }
}
