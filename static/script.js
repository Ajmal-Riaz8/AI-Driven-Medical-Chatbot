
document.addEventListener("DOMContentLoaded", function() {
    setTimeout(() => {
        document.querySelector(".preloader").style.display = "none";
    }, 2000);

    document.getElementById("askButton").addEventListener("click", function() {
        let userQuestion = document.getElementById("questionInput").value.trim();
        
        if (userQuestion === "") {
            alert("Please enter a question.");
            return;
        }

        fetch("/get_answer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: userQuestion })
        })
        .then(response => response.json())
        .then(data => {
            let responseField = document.getElementById("answerOutput");
            if (data.answer) {
                responseField.value = data.answer; // Correctly setting answer text
            } else {
                responseField.value = "Sorry, no response found."; // Fallback message
            }
        })
        .catch(error => {
            console.error("Error fetching the answer:", error);
            document.getElementById("answerOutput").value = "Error fetching the answer. Please try again.";
        });
    });
});
