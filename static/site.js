(async function updateHealth() {
  const dot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const lastCheck = document.getElementById("lastCheck");

  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();
    const ok = response.ok && data.status === "ok";
    dot.className = `dot ${ok ? "ok" : "error"}`;
    statusText.textContent = ok ? "Service healthy" : "Service degraded";
    lastCheck.textContent = data.last_check
      ? `Last check ${new Date(data.last_check * 1000).toLocaleString()}`
      : "Waiting for scheduler probe";
  } catch (error) {
    dot.className = "dot error";
    statusText.textContent = "Health unavailable";
    lastCheck.textContent = "Health endpoint did not respond";
  }
})();
