(function () {
    var chart = null;
    var playback = null;
    var timelineIndex = 0;
    var playbackTimer = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function setButtonLoading(button, loadingText, isLoading) {
        if (!button) return;
        if (!button.dataset.defaultText) button.dataset.defaultText = button.textContent;
        button.disabled = isLoading;
        button.textContent = isLoading ? loadingText : button.dataset.defaultText;
    }

    function getBandColor(band) {
        var colors = {
            theta: 'rgb(104, 187, 154)',
            alpha: 'rgb(99, 132, 255)',
            beta: 'rgb(255, 159, 64)'
        };
        return colors[band] || 'rgb(200, 200, 200)';
    }

    function getVisibleBands() {
        var visible = {};
        document.querySelectorAll('.eeg-bands input[name="band"]').forEach(function (checkbox) {
            visible[checkbox.value] = checkbox.checked;
        });
        return visible;
    }

    function quantile(values, q) {
        if (!values.length) return 0;
        var sorted = values.slice().sort(function (a, b) { return a - b; });
        var index = (sorted.length - 1) * q;
        var lower = Math.floor(index);
        var upper = Math.ceil(index);
        if (lower === upper) return sorted[lower];
        var weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    function smoothSeries(values, radius) {
        if (!values || values.length < 3 || radius <= 0) return values ? values.slice() : [];
        return values.map(function (_, index) {
            var start = Math.max(0, index - radius);
            var end = Math.min(values.length, index + radius + 1);
            var total = 0;
            for (var i = start; i < end; i += 1) total += values[i];
            return total / (end - start);
        });
    }

    function normalizeDisplaySeries(values) {
        if (!values || !values.length) return [];
        var smoothed = smoothSeries(values, 2);
        var absoluteValues = smoothed.map(function (value) { return Math.abs(value); });
        var clip = quantile(absoluteValues, 0.97) || 1;
        return smoothed.map(function (value) {
            var bounded = Math.max(-clip, Math.min(clip, value));
            return bounded / clip;
        });
    }

    function buildDatasets() {
        if (!playback) return [];
        var visibleBands = getVisibleBands();
        var displayOffsets = {
            theta: 2.4,
            alpha: 0,
            beta: -2.4
        };
        return ['theta', 'alpha', 'beta'].filter(function (band) {
            return visibleBands[band];
        }).map(function (band) {
            var normalized = normalizeDisplaySeries(playback[band]);
            return {
                label: band.charAt(0).toUpperCase() + band.slice(1),
                data: playback.time.map(function (timeValue, index) {
                    return { x: timeValue, y: normalized[index] + displayOffsets[band] };
                }),
                borderColor: getBandColor(band),
                backgroundColor: getBandColor(band).replace('rgb', 'rgba').replace(')', ', 0.16)'),
                borderWidth: 2.1,
                pointRadius: 0,
                tension: 0.28,
                fill: false
            };
        });
    }

    function ensureChart() {
        var canvas = byId('eegChart');
        if (!canvas || !canvas.getContext) return;
        var ctx = canvas.getContext('2d');
        if (!chart) {
            chart = new Chart(ctx, {
                type: 'line',
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    interaction: { intersect: false, mode: 'nearest' },
                    plugins: {
                        legend: {
                            labels: { color: '#d9def0' }
                        }
                    },
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Time (s)', color: '#9ba7c2' },
                            ticks: { color: '#9ba7c2' },
                            grid: { color: 'rgba(255,255,255,0.08)' }
                        },
                        y: {
                            min: -4,
                            max: 4,
                            title: { display: true, text: 'Band lanes', color: '#9ba7c2' },
                            ticks: {
                                color: '#9ba7c2',
                                callback: function (value) {
                                    if (value === 2.4) return 'Theta';
                                    if (value === 0) return 'Alpha';
                                    if (value === -2.4) return 'Beta';
                                    return '';
                                }
                            },
                            grid: {
                                color: function (context) {
                                    var laneValues = [2.4, 0, -2.4];
                                    return laneValues.indexOf(context.tick.value) >= 0
                                        ? 'rgba(255,255,255,0.16)'
                                        : 'rgba(255,255,255,0.03)';
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    function renderChart() {
        ensureChart();
        if (!chart || !playback) return;
        chart.data.datasets = buildDatasets();
        chart.update('none');
        updateCursor();
    }

    function updateCursor() {
        if (!playback || !playback.timeline.length || !chart) return;
        var cursor = byId('chartCursor');
        var wrapper = cursor ? cursor.parentElement : null;
        var current = playback.timeline[timelineIndex];
        var xScale = chart.scales && chart.scales.x;
        if (!cursor || !wrapper || !xScale || !chart.chartArea) return;
        var wrapperRect = wrapper.getBoundingClientRect();
        var canvasRect = chart.canvas.getBoundingClientRect();
        var left = (canvasRect.left - wrapperRect.left) + xScale.getPixelForValue(current.midpoint);
        cursor.style.left = String(left) + 'px';
        cursor.style.top = String((canvasRect.top - wrapperRect.top) + chart.chartArea.top) + 'px';
        cursor.style.height = String(chart.chartArea.bottom - chart.chartArea.top) + 'px';
    }

    function renderTimelineBar() {
        var container = byId('stateTimeline');
        if (!container || !playback) return;
        container.innerHTML = '';
        var lastTime = playback.time[playback.time.length - 1] || 1;
        playback.segments.forEach(function (segment) {
            var node = document.createElement('div');
            node.className = 'timeline-segment timeline-' + segment.label.toLowerCase();
            node.style.width = String(((segment.end - segment.start) / lastTime) * 100) + '%';
            node.title = segment.label + ' (' + Math.round(segment.confidence * 100) + '%)';
            container.appendChild(node);
        });
    }

    function setPulseState(label) {
        var pulse = byId('brainPulse');
        if (!pulse) return;
        pulse.classList.toggle('focused', label === 'Focused');
        pulse.classList.toggle('relaxed', label !== 'Focused');
    }

    function renderCurrentState() {
        var summaryEl = byId('stateSummary');
        var confidenceEl = byId('stateConfidence');
        var explanationEl = byId('stateExplanation');
        var disclaimerEl = byId('stateDisclaimer');
        var playbackMetaEl = byId('playbackMeta');
        if (!summaryEl || !confidenceEl || !explanationEl || !playbackMetaEl) return;

        if (!playback || !playback.timeline.length) {
            summaryEl.textContent = '—';
            confidenceEl.textContent = 'Confidence —';
            explanationEl.textContent = 'Train the first model, then load a session to see how the prediction changes over time.';
            playbackMetaEl.textContent = 'The state label will update as the cursor moves through the session.';
            if (disclaimerEl) disclaimerEl.style.display = 'none';
            return;
        }

        var current = playback.timeline[timelineIndex];
        summaryEl.textContent = current.label;
        confidenceEl.textContent = 'Confidence ' + Math.round(current.confidence * 100) + '%';
        explanationEl.textContent =
            current.label === 'Focused'
                ? 'The model sees a window that looks closer to the Shin2017 subtraction trials than the rest trials.'
                : 'The model sees a window that looks closer to the Shin2017 rest trials than the subtraction trials.';
        playbackMetaEl.textContent =
            'Window ' + (timelineIndex + 1) + ' of ' + playback.timeline.length +
            ' at ' + current.midpoint.toFixed(1) + 's in ' + playback.session_name + '.';
        if (disclaimerEl) disclaimerEl.style.display = 'block';
        setPulseState(current.label);
        updateCursor();
    }

    function stopPlayback() {
        if (playbackTimer) {
            window.clearInterval(playbackTimer);
            playbackTimer = null;
        }
        var button = byId('playPauseBtn');
        if (button) button.textContent = 'Play';
    }

    function setTimelineIndex(index) {
        if (!playback || !playback.timeline.length) return;
        timelineIndex = Math.max(0, Math.min(playback.timeline.length - 1, index));
        var slider = byId('timelineSlider');
        if (slider) slider.value = String(timelineIndex);
        renderCurrentState();
    }

    window.togglePlayback = function togglePlayback() {
        if (!playback || !playback.timeline.length) return;
        var button = byId('playPauseBtn');
        if (playbackTimer) {
            stopPlayback();
            return;
        }
        if (button) button.textContent = 'Pause';
        playbackTimer = window.setInterval(function () {
            if (!playback) {
                stopPlayback();
                return;
            }
            if (timelineIndex >= playback.timeline.length - 1) {
                stopPlayback();
                return;
            }
            setTimelineIndex(timelineIndex + 1);
        }, 350);
    };

    function applyPlayback(payload) {
        playback = payload;
        timelineIndex = 0;
        var slider = byId('timelineSlider');
        if (slider) {
            slider.max = String(Math.max(0, playback.timeline.length - 1));
            slider.value = '0';
        }
        var sessionMeta = byId('sessionMeta');
        if (sessionMeta) {
            sessionMeta.textContent =
                'Loaded subject ' + playback.subject + ', session ' + playback.session +
                ' (' + playback.session_name + '). Dominant state: ' + playback.summary.dominant_state + '.';
        }
        renderTimelineBar();
        renderChart();
        renderCurrentState();
        stopPlayback();
    }

    function handleApiError(prefix, error) {
        console.error(prefix, error);
        var message = error && error.message ? error.message : String(error);
        var status = byId('modelStatusText');
        if (status) status.textContent = prefix + ': ' + message;
    }

    function fetchJson(url, options) {
        return fetch(url, options).then(function (response) {
            if (!response.ok) {
                return response.json().catch(function () { return {}; }).then(function (payload) {
                    throw new Error(payload.detail || response.statusText || 'Request failed');
                });
            }
            return response.json();
        });
    }

    function refreshModelStatus() {
        return fetchJson('/api/model/status').then(function (payload) {
            var statusEl = byId('modelStatusText');
            if (!statusEl) return;
            if (!payload.trained) {
                statusEl.textContent = 'Model not trained yet. Train the first Shin2017 state model to unlock session playback.';
                return;
            }
            statusEl.textContent =
                'Model ready: ' + payload.metadata.model_name +
                ' on ' + payload.metadata.dataset +
                ' with test accuracy ' + Math.round(payload.metadata.test_accuracy * 100) + '%.';
        }).catch(function (error) {
            handleApiError('Model status failed', error);
        });
    }

    window.trainFirstModel = function trainFirstModel() {
        var button = byId('trainModelBtn');
        setButtonLoading(button, 'Training…', true);
        fetchJson('/api/model/train?model=svm', { method: 'POST' })
            .then(function (payload) {
                var statusEl = byId('modelStatusText');
                if (statusEl) {
                    statusEl.textContent =
                        'Training complete. Test accuracy ' + Math.round(payload.metrics.accuracy * 100) +
                        '%, cross-validation ' + Math.round(payload.metrics.cross_val_accuracy_mean * 100) + '%.';
                }
            })
            .catch(function (error) {
                handleApiError('Training failed', error);
            })
            .finally(function () {
                setButtonLoading(button, 'Training…', false);
            });
    };

    window.doLoadSession = function doLoadSession() {
        var button = byId('eegLoadBtn');
        var subject = parseInt(byId('eegSubject').value, 10) || 1;
        var session = parseInt(byId('eegSession').value, 10) || 1;
        setButtonLoading(button, 'Loading…', true);
        fetchJson('/api/session/playback?subject=' + subject + '&session=' + session)
            .then(applyPlayback)
            .catch(function (error) {
                handleApiError('Session load failed', error);
            })
            .finally(function () {
                setButtonLoading(button, 'Loading…', false);
            });
    };

    function bindControls() {
        document.querySelectorAll('.eeg-bands input[name="band"]').forEach(function (checkbox) {
            checkbox.addEventListener('change', renderChart);
        });
        var slider = byId('timelineSlider');
        if (slider) {
            slider.addEventListener('input', function (event) {
                stopPlayback();
                setTimelineIndex(parseInt(event.target.value, 10) || 0);
            });
        }
    }

    function init() {
        bindControls();
        refreshModelStatus();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
