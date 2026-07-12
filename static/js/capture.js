// Browser-based camera capture -> sends captured frame to /capture endpoint
(function () {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const startBtn = document.getElementById("startCamera");
  const captureBtn = document.getElementById("captureBtn");
  const overlay = document.getElementById("loadingOverlay");

  if (!video || !canvas || !startBtn || !captureBtn) return;

  let stream = null;

  startBtn.addEventListener("click", async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      video.srcObject = stream;
      captureBtn.disabled = false;
      startBtn.innerHTML = '<i class="fa-solid fa-check me-1"></i>Camera On';
      startBtn.disabled = true;
    } catch (err) {
      alert("Could not access the camera: " + err.message);
    }
  });

  captureBtn.addEventListener("click", async () => {
    if (!stream) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/png");

    if (overlay) overlay.classList.remove("d-none");
    captureBtn.disabled = true;

    try {
      const response = await fetch("/capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      const result = await response.json();

      if (result.success) {
        window.location.href = result.redirect;
      } else {
        if (overlay) overlay.classList.add("d-none");
        captureBtn.disabled = false;
        alert(result.message || "Failed to process the captured image.");
      }
    } catch (err) {
      if (overlay) overlay.classList.add("d-none");
      captureBtn.disabled = false;
      alert("Network error while sending the captured image.");
    }
  });
})();
