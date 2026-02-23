(function () {
    var chart = null;
    var eegData = null;

    function getColor(band) {
        var colors = { alpha: 'rgb(99, 132, 255)', beta: 'rgb(255, 159, 64)', gamma: 'rgb(75, 192, 192)', delta: 'rgb(153, 102, 255)' };
        return colors[band] || 'rgb(200, 200, 200)';
    }

    function buildDatasets(visibleBands) {
        if (!eegData) return [];
        var datasets = [];
        ['alpha', 'beta', 'gamma', 'delta'].forEach(function (band) {
            if (!visibleBands[band]) return;
            datasets.push({
                label: band.charAt(0).toUpperCase() + band.slice(1),
                data: eegData[band],
                borderColor: getColor(band),
                backgroundColor: getColor(band).replace('rgb', 'rgba').replace(')', ', 0.1)'),
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.2
            });
        });
        return datasets;
    }

    function updateChart() {
        var checkboxes = document.querySelectorAll('.eeg-bands input[name="band"]');
        var visible = {};
        checkboxes.forEach(function (cb) { visible[cb.value] = cb.checked; });
        if (!chart || !eegData) return;
        chart.data.datasets = buildDatasets(visible);
        chart.update('none');
    }

    function initChart() {
        var ctx = document.getElementById('eegChart').getContext('2d');
        var visible = { alpha: true, beta: true, gamma: true, delta: false };
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: eegData ? eegData.time : [],
                datasets: buildDatasets(visible)
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                scales: {
                    x: { display: true, title: { display: true, text: 'Time (s)' } },
                    y: { display: true, title: { display: true, text: 'Amplitude' } }
                }
            }
        });
    }

    function loadEegData() {
        fetch('/api/eeg/sample?seconds=10&sample_rate=128')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                eegData = data;
                if (!chart) initChart();
                else updateChart();
            })
            .catch(function (err) { console.error('EEG load failed', err); });
    }

    document.querySelectorAll('.eeg-bands input[name="band"]').forEach(function (cb) {
        cb.addEventListener('change', updateChart);
    });

    loadEegData();
})();
