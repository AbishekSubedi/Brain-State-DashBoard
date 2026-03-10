(function () {
    var chart = null;
    var eegData = null;

    window.doLoadEeg = function doLoadEeg() {
        var loadBtn = document.getElementById('eegLoadBtn');
        var subjectEl = document.getElementById('eegSubject');
        var sessionEl = document.getElementById('eegSession');
        var subject = subjectEl ? parseInt(subjectEl.value, 10) || 1 : 1;
        var session = sessionEl ? parseInt(sessionEl.value, 10) || 1 : 1;
        if (loadBtn) {
            loadBtn.disabled = true;
            loadBtn.textContent = 'Loading…';
        }
        var reenable = function () {
            if (loadBtn) {
                loadBtn.disabled = false;
                loadBtn.textContent = 'Load';
            }
        };
        var eegUrl = '/api/eeg/sample?subject=' + subject + '&session=' + session + '&max_duration_sec=60&sample_rate=128';
        var stateUrl = '/api/state?subject=' + subject + '&session=' + session + '&max_duration_sec=60';
        Promise.all([fetch(eegUrl).then(function (r) { if (!r.ok) throw new Error('EEG: ' + r.status); return r.json(); }), fetch(stateUrl).then(function (r) { if (!r.ok) throw new Error('State: ' + r.status); return r.json(); })])
            .then(function (results) {
                var data = results[0];
                var state = results[1];
                if (data && Array.isArray(data.time)) {
                    eegData = data;
                    if (typeof window.eegChartUpdate === 'function') window.eegChartUpdate();
                }
                if (state) renderState(state);
            })
            .catch(function (err) { console.error('Load failed', err); renderState(null); })
            .finally(reenable);
        setTimeout(reenable, 120000);
    };

    function renderState(state) {
        var summaryEl = document.getElementById('stateSummary');
        var explanationEl = document.getElementById('stateExplanation');
        var disclaimerEl = document.getElementById('stateDisclaimer');
        if (!summaryEl) return;
        if (!state) {
            summaryEl.textContent = '—';
            if (explanationEl) explanationEl.textContent = 'Load a session to see the inferred state.';
            if (disclaimerEl) disclaimerEl.style.display = 'none';
            return;
        }
        var label = state.predicted_state || '—';
        var conf = state.confidence != null ? Math.round(state.confidence * 100) + '%' : '';
        summaryEl.textContent = label + (conf ? ' (' + conf + ')' : '');
        if (explanationEl) explanationEl.textContent = state.explanation || '';
        if (disclaimerEl) disclaimerEl.style.display = 'block';
    }

    function runWhenReady(fn) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
        else fn();
    }

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
        var ctx = document.getElementById('eegChart');
        if (!ctx || !ctx.getContext) return;
        ctx = ctx.getContext('2d');
        var visible = { alpha: true, beta: true, gamma: true, delta: false };
        chart = new Chart(ctx, {
            type: 'line',
            data: { labels: eegData ? eegData.time : [], datasets: buildDatasets(visible) },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                scales: { x: { display: true, title: { display: true, text: 'Time (s)' } }, y: { display: true, title: { display: true, text: 'Amplitude' } } }
            }
        });
    }

    window.eegChartUpdate = function () {
        if (!eegData) return;
        if (!chart) initChart();
        else updateChart();
    };

    function loadInitial() {
        var subjectEl = document.getElementById('eegSubject');
        var sessionEl = document.getElementById('eegSession');
        var subject = subjectEl ? parseInt(subjectEl.value, 10) || 1 : 1;
        var session = sessionEl ? parseInt(sessionEl.value, 10) || 1 : 1;
        var eegUrl = '/api/eeg/sample?subject=' + subject + '&session=' + session + '&max_duration_sec=60&sample_rate=128';
        var stateUrl = '/api/state?subject=' + subject + '&session=' + session + '&max_duration_sec=60';
        Promise.all([
            fetch(eegUrl).then(function (r) { return r.ok ? r.json() : null; }),
            fetch(stateUrl).then(function (r) { return r.ok ? r.json() : null; })
        ]).then(function (results) {
            if (results[0] && Array.isArray(results[0].time)) {
                eegData = results[0];
                if (!chart) initChart();
                else updateChart();
            }
            renderState(results[1] || null);
        }).catch(function () { renderState(null); });
    }

    function init() {
        document.querySelectorAll('.eeg-bands input[name="band"]').forEach(function (cb) {
            cb.addEventListener('change', updateChart);
        });
        loadInitial();
    }

    runWhenReady(init);
})();
