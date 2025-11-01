    document.getElementById("trainForm").addEventListener("submit", async function (e) {
  e.preventDefault(); // prevent normal form submit

  const selectedAction = document.getElementById("action").value;
  document.getElementById("result").innerText = "Training in progress...";

  try {
    const response = await fetch("http://127.0.0.1:8000/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ action: selectedAction })
    });

    const data = await response.json();
    document.getElementById("result").innerText = 
      `Training complete!\nAction: ${data.action}\nFinal Loss: ${data.final_loss}`;
  } catch (error) {
    document.getElementById("result").innerText = "Error: " + error;
  }
});
