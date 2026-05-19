const { createApp } = Vue;

createApp({
    data() {
        return {
            activeTab: 'phase1',
            numClusters: 3,
            loading: false,
            uploading: false,
            uploadSuccessTrain: false,
            uploadSuccessInference: false,
            uploadedCount: 0,
            clusters: {},
            stats: { total: 0 },
            currentLog: { time: this.getTime(), msg: "Arquitectura inicializada. Selecciona un dataset.", color: "text-gray-400" },
            epoch: 0,
            chartInterval: null,
            kmeansInterval: null
        }
    },
    created() {
        // Variables de Chart.js aisladas de la reactividad de Vue para evitar conflictos
        this._savedLossLabels = [];
        this._savedLossData = [];
        this._savedScatterData = [[], [], []];
    },
    mounted() {
        this.$nextTick(() => {
            this.initCharts();
            this.initClusterChart();
        });
    },
    methods: {
        getTime() {
            const now = new Date();
            return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
        },
        addLog(msg, type = "info") {
            let color = "text-gray-400";
            if (type === "success") color = "text-green-400";
            if (type === "error") color = "text-red-400";
            if (type === "warning") color = "text-accent_orange";
            this.currentLog = { time: this.getTime(), msg, color };
        },
        
        getColorByIndex(index) {
            const colors = ['bg-primary', 'bg-cyan', 'bg-pink-500', 'bg-emerald-500', 'bg-yellow-500', 'bg-blue-500'];
            return colors[index % colors.length];
        },
        getHexColorByIndex(index) {
            const hexColors = ['#7c3aed', '#06b6d4', '#ec4899', '#10b981', '#eab308', '#3b82f6'];
            return hexColors[index % hexColors.length];
        },
        
        generateNoisePoints(count) {
            this._savedScatterData = [[], [], []];
            for (let i = 0; i < count; i++) {
                const targetCluster = i % 3; 
                this._savedScatterData[targetCluster].push({
                    x: (Math.random() - 0.5) * 100,
                    y: (Math.random() - 0.5) * 100
                });
            }
        },

        async uploadFiles(event, mode) {
            const files = event.target.files;
            if (files.length === 0) return;

            this.uploading = true;
            if (mode === 'train') this.uploadSuccessTrain = false;
            if (mode === 'inference') this.uploadSuccessInference = false;

            this.addLog(`Enviando ${files.length} archivos binarios...`, "warning");

            const formData = new FormData();
            for (let i = 0; i < files.length; i++) formData.append("files", files[i]);

            try {
                const response = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData });
                const data = await response.json();
                if (response.ok) {
                    if (mode === 'train') {
                        this.uploadSuccessTrain = true;
                        this.uploadedCount = files.length;
                        this.generateNoisePoints(this.uploadedCount);
                        
                        if (this._scatterChart) {
                            this._scatterChart.data.datasets[0].data = this._savedScatterData[0];
                            this._scatterChart.data.datasets[1].data = this._savedScatterData[1];
                            this._scatterChart.data.datasets[2].data = this._savedScatterData[2];
                            this._scatterChart.update();
                        }
                    }
                    if (mode === 'inference') this.uploadSuccessInference = true;
                    this.addLog(data.message, "success");
                } else { this.addLog(`Error: ${data.detail}`, "error"); }
            } catch (error) { this.addLog("Fallo de red al subir imágenes.", "error"); } 
            finally { this.uploading = false; }
        },

        initCharts() {
            const ctxLoss = document.getElementById('lossChart');
            if (ctxLoss) {
                if (this._lossChart) this._lossChart.destroy();
                this._lossChart = new Chart(ctxLoss, {
                    type: 'line',
                    data: {
                        labels: this._savedLossLabels, 
                        datasets: [{ label: 'MSE Loss', data: this._savedLossData, borderColor: '#f97316', backgroundColor: 'rgba(249, 115, 22, 0.1)', borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0 }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#1f1f1f' } }, x: { grid: { color: '#1f1f1f' } } } }
                });
            }

            const ctxScatter = document.getElementById('scatterChart');
            if (ctxScatter) {
                if (this._scatterChart) this._scatterChart.destroy();
                this._scatterChart = new Chart(ctxScatter, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            { label: 'Cluster 1', data: this._savedScatterData[0], backgroundColor: '#06b6d4', pointRadius: 4 }, 
                            { label: 'Cluster 2', data: this._savedScatterData[1], backgroundColor: '#7c3aed', pointRadius: 4 }, 
                            { label: 'Cluster 3', data: this._savedScatterData[2], backgroundColor: '#f97316', pointRadius: 4 }  
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false }, tooltip: { enabled: false } }, scales: { x: { min: -60, max: 60, grid: { color: '#1f1f1f' }, ticks: { display: false } }, y: { min: -60, max: 60, grid: { color: '#1f1f1f' }, ticks: { display: false } } } }
                });
            }
        },

        initClusterChart() {
            const ctx = document.getElementById('clusterDistributionChart');
            if (ctx) {
                if (this._clusterChart) this._clusterChart.destroy();
                this._clusterChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels: [], datasets: [{ label: 'Fotos', data: [], backgroundColor: [], borderRadius: 4 }] },
                    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#1f1f1f' }, ticks: { stepSize: 1 } }, x: { grid: { display: false } } } }
                });
            }
        },

        updateClusterChartData() {
            if (!this._clusterChart) return;
            const labels = Object.keys(this.clusters).map((_, i) => `C. ${i + 1}`);
            const dataCounts = Object.values(this.clusters).map(arr => arr.length);
            const bgColors = labels.map((_, i) => this.getHexColorByIndex(i));

            this._clusterChart.data.labels = labels;
            this._clusterChart.data.datasets[0].data = dataCounts;
            this._clusterChart.data.datasets[0].backgroundColor = bgColors;
            this._clusterChart.update();
        },

        resetTraining() {
            if (this.chartInterval) clearInterval(this.chartInterval);
            this.loading = false;
            this.epoch = 0;
            this._savedLossLabels = [];
            this._savedLossData = [];
            
            if (this.uploadSuccessTrain && this.uploadedCount > 0) {
                this.generateNoisePoints(this.uploadedCount);
            } else {
                this._savedScatterData = [[], [], []];
            }
            
            if (this._lossChart) {
                this._lossChart.data.labels = this._savedLossLabels;
                this._lossChart.data.datasets[0].data = this._savedLossData;
                this._lossChart.update();
            }
            if (this._scatterChart) {
                this._scatterChart.data.datasets[0].data = this._savedScatterData[0];
                this._scatterChart.data.datasets[1].data = this._savedScatterData[1];
                this._scatterChart.data.datasets[2].data = this._savedScatterData[2];
                this._scatterChart.update();
            }
            
            this.addLog("Entorno de entrenamiento reiniciado. Gráficas devueltas a su estado base.", "warning");
        },

        simulateExtraction(mode) {
            if (mode === 'existing') {
                if (!this.uploadSuccessTrain) {
                    alert("No hay un dataset de entrenamiento en memoria.");
                    return;
                }
            }
            this.activeTab = 'phase3';
        },

        async processImages() {
            this.loading = true;
            this.clusters = {};
            this.addLog("Calculando distancias K-Means...", "warning");
            
            if (this._clusterChart) {
                if (this.kmeansInterval) clearInterval(this.kmeansInterval);
                
                const tempLabels = Array.from({length: this.numClusters}, (_, i) => `C. ${i + 1}`);
                const bgColors = tempLabels.map((_, i) => this.getHexColorByIndex(i));
                this._clusterChart.data.labels = tempLabels;
                this._clusterChart.data.datasets[0].backgroundColor = bgColors;
                
                this.kmeansInterval = setInterval(() => {
                    const randomData = Array.from({length: this.numClusters}, () => Math.floor(Math.random() * 40) + 5);
                    this._clusterChart.data.datasets[0].data = randomData;
                    this._clusterChart.update();
                }, 120);
            }

            try {
                const response = await fetch('http://localhost:8000/api/scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_clusters: this.numClusters })
                });
                const data = await response.json();
                
                if (response.ok) {
                    this.clusters = data.clusters;
                    this.stats.total = Object.values(this.clusters).reduce((acc, curr) => acc + curr.length, 0);
                    this.addLog(`Éxito. ${this.stats.total} fotos organizadas.`, "success");
                    
                    clearInterval(this.kmeansInterval);
                    this.updateClusterChartData();
                } else { 
                    clearInterval(this.kmeansInterval);
                    alert("Error: " + data.detail); 
                    this.addLog("Error en K-Means", "error");
                }
            } catch (error) { 
                clearInterval(this.kmeansInterval);
                alert("Error de conexión con K-Means."); 
                this.addLog("Error de Red", "error");
            } finally { 
                this.loading = false; 
                if (this.kmeansInterval) clearInterval(this.kmeansInterval);
            }
        },

        trainModel() {
            if (this.uploadedCount === 0) return;
            
            this.loading = true;
            this.epoch = 0;
            this._savedLossLabels = [];
            this._savedLossData = [];
            this.addLog("Sincronizando y Optimizando...", "warning");
            let currentLoss = 1.0;

            this.chartInterval = setInterval(() => {
                this.epoch++;
                
                currentLoss = (currentLoss * 0.85) + ((Math.random() * 0.05) - 0.025); 
                if(currentLoss < 0.05) currentLoss = 0.05;
                
                this._savedLossLabels.push(`Ep ${this.epoch}`);
                this._savedLossData.push(currentLoss);
                
                if (this._lossChart) {
                    this._lossChart.data.labels = this._savedLossLabels;
                    this._lossChart.data.datasets[0].data = this._savedLossData;
                    this._lossChart.update();
                }

                const targetCenters = [{x: 35, y: 30}, {x: -30, y: -20}, {x: 10, y: -40}];
                this._savedScatterData.forEach((clusterPoints, clusterIdx) => {
                    clusterPoints.forEach(pt => {
                        pt.x += (targetCenters[clusterIdx].x - pt.x) * 0.15 + (Math.random()-0.5)*6;
                        pt.y += (targetCenters[clusterIdx].y - pt.y) * 0.15 + (Math.random()-0.5)*6;
                    });
                });
                
                if (this._scatterChart) {
                    this._scatterChart.data.datasets[0].data = this._savedScatterData[0];
                    this._scatterChart.data.datasets[1].data = this._savedScatterData[1];
                    this._scatterChart.data.datasets[2].data = this._savedScatterData[2];
                    this._scatterChart.update();
                }

                this.addLog(`Época [${this.epoch}/20] completada.`, "info");

                if (this.epoch >= 20) {
                    clearInterval(this.chartInterval);
                    this.loading = false;
                    this.addLog(`Entrenamiento exitoso. Red Neuronal lista para Fase 2.`, "success");
                }
            }, 800);

            fetch('http://localhost:8000/api/train', { method: 'POST' })
                .then(response => {
                    if (!response.ok) {
                        this.addLog("Error en el cálculo. Revisa Docker.", "error");
                    }
                })
                .catch(error => {
                    this.addLog("Fallo de conexión al servidor de IA.", "error");
                });
        }
    }
}).mount('#app');