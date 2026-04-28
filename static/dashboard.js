(function () {
    var chart = null;
    var playback = null;
    var timelineIndex = 0;
    var playbackTimer = null;
    var GRAPH_WINDOW_SECONDS = 2.4;
    var activeMode = 'state';
    var MODEL_CONFIG = {
        state: {
            headline: 'Shin2017 First Model',
            intro: "This version of the dashboard uses the Shin2017 study as the first training source. For the current model, the usable state labels come from the study\\'s mental arithmetic split: rest = relaxed and subtraction = focused. Train the model, choose a subject and session, then watch the EEG bands and predicted state move through the session timeline.",
            statusUrl: '/api/model/status',
            trainUrl: '/api/model/train?model=svm',
            playbackUrl: '/api/session/playback',
            sessionsUrl: '/api/sessions?kind=state',
            trainButtonLabel: 'Train First Model',
            loadHint: 'Choose a subject and arithmetic session to load playback data.',
            statePanelTitle: 'State of Mind',
            simulationTitle: 'Playback Simulation',
            simulationIdle: 'Idle',
            simulationCaption: 'The simulation shifts between calm and engaged states as the current prediction changes.',
            summary: {
                dataset: 'Shin2017B',
                classes: 'Relaxed, Focused',
                baseline: 'Bandpower + SVM',
                note: 'Playback uses a sliding EEG window so the graph and the current prediction stay in sync.'
            },
            legend: [
                { label: 'Relaxed', swatchClass: 'swatch-relaxed' },
                { label: 'Focused', swatchClass: 'swatch-focused' }
            ],
            summaryNoun: 'state',
            simulationClass: function (label) {
                return label === 'Focused' ? 'state-focused' : 'state-relaxed';
            }
        },
        imagery: {
            headline: 'Shin2017 Second Model',
            intro: 'The second model uses Shin2017A motor-imagery trials to classify left-hand versus right-hand intent. Train the imagery model, choose a subject and session, then step through the EEG playback to see the predicted movement side change over time.',
            statusUrl: '/api/model/imagery/status',
            trainUrl: '/api/model/imagery/train?model=csp_lda',
            playbackUrl: '/api/session/imagery/playback',
            sessionsUrl: '/api/sessions?kind=imagery',
            trainButtonLabel: 'Train Second Model',
            loadHint: 'Choose a subject and imagery session to load left-vs-right playback data.',
            statePanelTitle: 'Predicted Movement',
            simulationTitle: 'Intent Simulation',
            simulationIdle: 'Awaiting intent',
            simulationCaption: 'The avatar raises the arm that matches the current left-vs-right imagery prediction and highlights the active target.',
            summary: {
                dataset: 'Shin2017A',
                classes: 'Left Hand, Right Hand',
                baseline: 'Filter-Bank CSP + LDA',
                note: 'Imagery playback uses the second model to decode left-vs-right hand intent over sliding windows.'
            },
            legend: [
                { label: 'Left Hand', swatchClass: 'swatch-left' },
                { label: 'Right Hand', swatchClass: 'swatch-right' }
            ],
            summaryNoun: 'movement',
            simulationClass: function (label) {
                return label === 'Right Hand' ? 'intent-right' : 'intent-left';
            }
        }
    };

    function byId(id) {
        return document.getElementById(id);
    }

    function getModeConfig() {
        return MODEL_CONFIG[activeMode];
    }

    function setButtonLoading(button, loadingText, isLoading) {
        if (!button) return;
        if (!button.dataset.defaultText) button.dataset.defaultText = button.textContent;
        button.disabled = isLoading;
        button.textContent = isLoading ? loadingText : button.dataset.defaultText;
    }

    function setText(id, value) {
        var node = byId(id);
        if (node) node.textContent = value;
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
                            min: 0,
                            max: GRAPH_WINDOW_SECONDS,
                            title: { display: true, text: 'Time (s)', color: '#9ba7c2' },
                            ticks: {
                                color: '#9ba7c2',
                                stepSize: 0.1,
                                maxRotation: 0,
                                minRotation: 0,
                                callback: function (value) {
                                    return Number(value).toFixed(1);
                                }
                            },
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
        if (!chart) return;
        if (!playback) {
            chart.data.datasets = [];
            chart.options.scales.x.min = 0;
            chart.options.scales.x.max = GRAPH_WINDOW_SECONDS;
            chart.update('none');
            return;
        }
        chart.data.datasets = buildDatasets();
        updateChartViewport();
        chart.update('none');
    }

    function updateChartViewport() {
        if (!chart || !playback || !playback.time.length) return;
        var current = playback.timeline && playback.timeline.length ? playback.timeline[timelineIndex] : null;
        var currentTime = current ? current.midpoint : 0;
        var sessionEnd = playback.time[playback.time.length - 1] || GRAPH_WINDOW_SECONDS;
        var halfWindow = GRAPH_WINDOW_SECONDS / 2;
        var minTime = Math.max(0, currentTime - halfWindow);
        var maxTime = minTime + GRAPH_WINDOW_SECONDS;

        if (maxTime > sessionEnd) {
            maxTime = sessionEnd;
            minTime = Math.max(0, maxTime - GRAPH_WINDOW_SECONDS);
        }

        chart.options.scales.x.min = minTime;
        chart.options.scales.x.max = maxTime;
    }

    function renderTimelineBar() {
        var container = byId('stateTimeline');
        if (!container) return;
        container.innerHTML = '';
        if (!playback) return;
        var lastTime = playback.time[playback.time.length - 1] || 1;
        playback.segments.forEach(function (segment) {
            var node = document.createElement('div');
            node.className = 'timeline-segment timeline-' + segment.label.toLowerCase().replace(/\s+/g, '-');
            node.style.width = String(((segment.end - segment.start) / lastTime) * 100) + '%';
            node.title = segment.label + ' (' + Math.round(segment.confidence * 100) + '%)';
            container.appendChild(node);
        });
    }

    function updateSimulation(label) {
        var scene = byId('simulationScene');
        var intent = byId('simulationIntent');
        var config = getModeConfig();
        if (!scene || !intent) return;
        scene.classList.remove(
            'state-relaxed',
            'state-focused',
            'intent-left',
            'intent-right',
            'is-idle',
            'target-left-active',
            'target-right-active'
        );
        if (!label) {
            scene.classList.add('is-idle');
            intent.textContent = config.simulationIdle;
            return;
        }
        var sceneClass = config.simulationClass(label);
        scene.classList.add(sceneClass);
        if (sceneClass === 'intent-left') {
            scene.classList.add('target-left-active');
        } else if (sceneClass === 'intent-right') {
            scene.classList.add('target-right-active');
        }
        intent.textContent = label;
    }

    function renderCurrentState() {
        var summaryEl = byId('stateSummary');
        var confidenceEl = byId('stateConfidence');
        var disclaimerEl = byId('stateDisclaimer');
        var playbackMetaEl = byId('playbackMeta');
        var currentLabelBadge = byId('currentLabelBadge');
        var dominantLabelBadge = byId('dominantLabelBadge');
        if (!summaryEl || !confidenceEl || !playbackMetaEl) return;

        if (!playback || !playback.timeline.length) {
            summaryEl.textContent = '—';
            confidenceEl.textContent = 'Confidence —';
            playbackMetaEl.textContent = 'The prediction label will update as playback moves through the session.';
            if (currentLabelBadge) currentLabelBadge.textContent = '—';
            if (dominantLabelBadge) dominantLabelBadge.textContent = '—';
            if (disclaimerEl) disclaimerEl.style.display = 'none';
            updateSimulation(null);
            return;
        }

        var current = playback.timeline[timelineIndex];
        var displayTime = Math.max(0, current.start);
        summaryEl.textContent = current.label;
        confidenceEl.textContent = 'Confidence ' + Math.round(current.confidence * 100) + '%';
        playbackMetaEl.textContent =
            'Window ' + (timelineIndex + 1) + ' of ' + playback.timeline.length +
            ' at ' + displayTime.toFixed(1) + 's in ' + playback.session_name + '.';
        if (currentLabelBadge) currentLabelBadge.textContent = current.label;
        if (dominantLabelBadge) dominantLabelBadge.textContent = playback.summary.dominant_label;
        if (disclaimerEl) disclaimerEl.style.display = 'block';
        updateSimulation(current.label);
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
        updateChartViewport();
        if (chart) chart.update('none');
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

    window.stepPlayback = function stepPlayback(delta) {
        if (!playback || !playback.timeline.length) return;
        stopPlayback();
        setTimelineIndex(timelineIndex + delta);
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
                ' (' + playback.session_name + '). Dominant ' + getModeConfig().summaryNoun + ': ' + playback.summary.dominant_label + '.';
        }
        renderTimelineBar();
        renderChart();
        renderCurrentState();
        stopPlayback();
    }

    function populateSessionOptions(payload) {
        var select = byId('sessionSelect');
        var pickerMeta = byId('sessionPickerMeta');
        if (!select) return;
        select.innerHTML = '';
        if (!payload.sessions || !payload.sessions.length) {
            var emptyOption = document.createElement('option');
            emptyOption.value = '1';
            emptyOption.textContent = 'No sessions found';
            select.appendChild(emptyOption);
            if (pickerMeta) pickerMeta.textContent = 'No sessions were found for this subject.';
            return;
        }
        payload.sessions.forEach(function (item, index) {
            var option = document.createElement('option');
            option.value = String(item.session);
            option.textContent = 'Session ' + item.session + ' · ' + item.session_name;
            if (index === 0) option.selected = true;
            select.appendChild(option);
        });
        if (pickerMeta) {
            pickerMeta.textContent =
                'Found ' + payload.sessions.length + ' ' + (payload.sessions.length === 1 ? 'session' : 'sessions') +
                ' in ' + payload.dataset + ' for subject ' + payload.subject + '.';
        }
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
        var config = getModeConfig();
        return fetchJson(config.statusUrl).then(function (payload) {
            var statusEl = byId('modelStatusText');
            if (!statusEl) return;
            if (!payload.trained) {
                statusEl.textContent = 'Model not trained yet. Train the selected Shin2017 model to unlock session playback.';
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

    window.refreshSessionOptions = function refreshSessionOptions() {
        var subject = parseInt(byId('eegSubject').value, 10) || 1;
        var button = byId('refreshSessionsBtn');
        var pickerMeta = byId('sessionPickerMeta');
        setButtonLoading(button, 'Finding…', true);
        if (pickerMeta) pickerMeta.textContent = 'Looking up available sessions…';
        return fetchJson(getModeConfig().sessionsUrl + '&subject=' + subject)
            .then(populateSessionOptions)
            .catch(function (error) {
                if (pickerMeta) pickerMeta.textContent = 'Unable to load sessions right now.';
                handleApiError('Session lookup failed', error);
            })
            .finally(function () {
                setButtonLoading(button, 'Finding…', false);
            });
    };

    window.trainActiveModel = function trainActiveModel() {
        var button = byId('trainModelBtn');
        setButtonLoading(button, 'Training…', true);
        fetchJson(getModeConfig().trainUrl, { method: 'POST' })
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
        var session = parseInt(byId('sessionSelect').value, 10) || 1;
        setButtonLoading(button, 'Loading…', true);
        fetchJson(getModeConfig().playbackUrl + '?subject=' + subject + '&session=' + session)
            .then(applyPlayback)
            .catch(function (error) {
                handleApiError('Session load failed', error);
            })
            .finally(function () {
                setButtonLoading(button, 'Loading…', false);
            });
    };

    function updateModeUI() {
        var config = getModeConfig();
        var headline = byId('modelHeadline');
        var intro = byId('modelIntro');
        var trainButton = byId('trainModelBtn');
        var sessionMeta = byId('sessionMeta');
        var legendLabelA = byId('legendLabelA');
        var legendLabelB = byId('legendLabelB');
        var legendSwatchA = byId('legendSwatchA');
        var legendSwatchB = byId('legendSwatchB');
        var statePanelTitle = byId('statePanelTitle');
        var simulationTitle = byId('simulationTitle');
        var simulationCaption = byId('simulationCaption');

        document.querySelectorAll('.model-chip').forEach(function (chip) {
            chip.classList.toggle('active', chip.dataset.modelMode === activeMode);
        });

        if (headline) headline.textContent = config.headline;
        if (intro) intro.textContent = config.intro;
        if (trainButton) {
            trainButton.textContent = config.trainButtonLabel;
            trainButton.dataset.defaultText = config.trainButtonLabel;
        }
        if (sessionMeta) sessionMeta.textContent = config.loadHint;
        if (legendLabelA) legendLabelA.textContent = config.legend[0].label;
        if (legendLabelB) legendLabelB.textContent = config.legend[1].label;
        if (legendSwatchA) legendSwatchA.className = 'swatch ' + config.legend[0].swatchClass;
        if (legendSwatchB) legendSwatchB.className = 'swatch ' + config.legend[1].swatchClass;
        if (statePanelTitle) statePanelTitle.textContent = config.statePanelTitle;
        if (simulationTitle) simulationTitle.textContent = config.simulationTitle;
        if (simulationCaption) simulationCaption.textContent = config.simulationCaption;
        setText('summaryDataset', config.summary.dataset);
        setText('summaryClasses', config.summary.classes);
        setText('summaryBaseline', config.summary.baseline);
        setText('summaryNote', config.summary.note);

        playback = null;
        stopPlayback();
        updateSimulation(null);
        renderCurrentState();
        renderTimelineBar();
        renderChart();
        refreshModelStatus();
        window.refreshSessionOptions();
    }

    function bindControls() {
        document.querySelectorAll('.eeg-bands input[name="band"]').forEach(function (checkbox) {
            checkbox.addEventListener('change', renderChart);
        });
        document.querySelectorAll('.model-chip').forEach(function (chip) {
            chip.addEventListener('click', function () {
                if (chip.dataset.modelMode === activeMode) return;
                activeMode = chip.dataset.modelMode;
                updateModeUI();
            });
        });
        var slider = byId('timelineSlider');
        if (slider) {
            slider.addEventListener('input', function (event) {
                stopPlayback();
                setTimelineIndex(parseInt(event.target.value, 10) || 0);
            });
        }
        var subjectInput = byId('eegSubject');
        if (subjectInput) {
            subjectInput.addEventListener('change', function () {
                window.refreshSessionOptions();
            });
        }
    }

    function init() {
        bindControls();
        updateModeUI();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
