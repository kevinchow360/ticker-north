function getMetrics() {
    const ticker = document.getElementById("ticker").value.toUpperCase();
    const url = `/api/ticker/${ticker}`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            document.getElementById("results").textContent = JSON.stringify(data, null, 2);
        })
        .catch(err => {
            document.getElementById("results").textContent = "Error: " + err;
        });
}
