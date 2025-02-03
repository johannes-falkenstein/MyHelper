async function sendMessage() {
    const prompt = document.getElementById("prompt").value.trim();
    const model = document.getElementById("model").value;
    const messagesDiv = document.getElementById("messages");

    if (prompt === "") return; // Don't send empty messages

    // Add user message to the chat
    const userMessage = document.createElement("div");
    userMessage.className = "message user-message";
    userMessage.innerText = prompt;
    messagesDiv.appendChild(userMessage);

    // Clear input
    document.getElementById("prompt").value = "";

    // Scroll to latest message
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, model })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const result = await response.json();
        let aiResponse = result.response || "Error: Unexpected response format";

        // Process <think> section
        const thinkRegex = /<think>([\s\S]*?)<\/think>/;
        const thinkMatch = aiResponse.match(thinkRegex);

        let thinkContent = "";
        if (thinkMatch) {
            thinkContent = thinkMatch[1];
            aiResponse = aiResponse.replace(
                thinkRegex,
                `<div class="think-container">
                    <button class="show-think" onclick="toggleThink(this)">Show Thinking Process</button>
                    <div class="think-content" style="display: none;">${thinkContent}</div>
                 </div>`
            );
        }

        // Convert Markdown to HTML using Marked.js
        const aiMessage = document.createElement("div");
        aiMessage.className = "message ai-message";
        aiMessage.innerHTML = marked.parse(aiResponse); // Render Markdown properly
        messagesDiv.appendChild(aiMessage);

        // Scroll to the latest message
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

    } catch (error) {
        console.error("Fetch error:", error);
        const errorMessage = document.createElement("div");
        errorMessage.className = "message ai-message";
        errorMessage.innerText = "Failed to connect to AI server.";
        messagesDiv.appendChild(errorMessage);
    }
}

// Function to toggle the visibility of the think process
function toggleThink(button) {
    const thinkContainer = button.parentElement; // Get the parent container
    const thinkContent = thinkContainer.querySelector(".think-content"); // Find the hidden think content

    if (thinkContent) {
        if (thinkContent.style.display === "none") {
            thinkContent.style.display = "block";
            button.innerText = "Hide Thinking Process";
        } else {
            thinkContent.style.display = "none";
            button.innerText = "Show Thinking Process";
        }
    } else {
        console.error("Think content not found!");
    }
}