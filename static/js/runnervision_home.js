document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    let uploadedFiles = [];
    
    // Setup file input and drop zone
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', () => {
            uploadedFiles = Array.from(fileInput.files);
            dropZone.querySelector('p').textContent = `${uploadedFiles.length} file(s) ready to upload`;
        });
        
        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            uploadedFiles = Array.from(e.dataTransfer.files);
            dropZone.querySelector('p').textContent = `${uploadedFiles.length} file(s) ready to upload`;
        });
    }
    
    // Setup run analysis button
    const runButton = document.getElementById('run-btn');
    if (runButton) {
        runButton.addEventListener('click', startAnalysis);
    }

    // Analysis function
    async function startAnalysis() {
        if (!uploadedFiles.length) {
            alert('Please upload 1 or 2 video files first.');
            return;
        }
        
        const overlay = document.getElementById('overlay');
        overlay.style.display = 'flex';
        
        let progress = 0;
        const bar = document.getElementById('progress-bar-inner');
        const interval = setInterval(() => {
            if (progress < 90) {
                progress += 10;
                bar.style.width = progress + '%';
            }
        }, 300);
        
        try {
            // Upload the files
            const formData = new FormData();
            uploadedFiles.forEach(file => formData.append('videos', file));
            
            const uploadResp = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!uploadResp.ok) {
                throw new Error('Upload failed');
            }
            
            // Run the analysis
            const analysisResp = await fetch('/run_biomechanic_analysis', {
                method: 'POST',
            });
            if (!analysisResp.ok) {
                throw new Error('Analysis request failed');
            }
            
            const data = await analysisResp.json();
            
            // Complete the progress bar
            clearInterval(interval);
            bar.style.width = '100%';
            
            // Show results
            if (data.status === 'complete') {
                const rearVideo = document.getElementById('rear-video');
                const rearReport = document.getElementById('rear-report');
                const sideVideo = document.getElementById('side-video');
                const sideReport = document.getElementById('side-report');
                
                // Set video sources and report links
                if (data.rear_video_path) {
                    rearVideo.src = data.rear_video_path;
                    rearVideo.type = "video/mp4"; // Explicitly set the MIME type
                    
                    // Force video reload
                    rearVideo.load();
                    
                    // Set report link if available
                    if (data.rear_report_path) {
                        rearReport.href = data.rear_report_path;
                        // Open in iframe instead of new tab
                        rearReport.setAttribute('data-report', data.rear_report_path);
                        rearReport.addEventListener('click', function(e) {
                            e.preventDefault();
                            openReportInOverlay(data.rear_report_path);
                        });
                    }
                }
                
                if (data.side_video_path) {
                    sideVideo.src = data.side_video_path;
                    sideVideo.type = "video/mp4"; // Explicitly set the MIME type
                    
                    // Force video reload
                    sideVideo.load();
                    
                    // Set report link if available
                    if (data.side_report_path) {
                        sideReport.href = data.side_report_path;
                        // Open in iframe instead of new tab
                        sideReport.setAttribute('data-report', data.side_report_path);
                        sideReport.addEventListener('click', function(e) {
                            e.preventDefault();
                            openReportInOverlay(data.side_report_path);
                        });
                    }
                }
                
                document.getElementById('results').style.display = 'flex';
            } else {
                throw new Error('Analysis failed');
            }
            
        } catch (error) {
            alert(error.message || 'An error occurred');
        } finally {
            // Hide overlay after a short delay
            setTimeout(() => {
                overlay.style.display = 'none';
                if (bar) bar.style.width = '0';
            }, 1000);
        }
    }
    
    // Function to open reports in overlay
    function openReportInOverlay(reportUrl) {
        // Create overlay if it doesn't exist
        let reportOverlay = document.getElementById('report-overlay');
        if (!reportOverlay) {
            reportOverlay = document.createElement('div');
            reportOverlay.id = 'report-overlay';
            reportOverlay.className = 'report-overlay';
            
            // Add close button
            const closeBtn = document.createElement('button');
            closeBtn.textContent = '×';
            closeBtn.className = 'close-report-btn';
            closeBtn.addEventListener('click', () => {
                reportOverlay.style.display = 'none';
            });
            
            // Add iframe
            const iframe = document.createElement('iframe');
            iframe.id = 'report-iframe';
            iframe.className = 'report-iframe';
            
            reportOverlay.appendChild(closeBtn);
            reportOverlay.appendChild(iframe);
            document.body.appendChild(reportOverlay);
        }
        
        // Set iframe source and show overlay
        const iframe = document.getElementById('report-iframe');
        iframe.src = reportUrl;
        reportOverlay.style.display = 'flex';
    }
});