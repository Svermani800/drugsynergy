const $ = (id) => document.getElementById(id);
const form = $('prediction-form');

function fillSelect(select, values) {
  select.innerHTML = values.map(value => `<option value="${value}">${value}</option>`).join('');
}

async function loadMetadata() {
  const response = await fetch('/api/metadata');
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || 'Model is unavailable');
  fillSelect($('drug-a'), data.drugs);
  fillSelect($('drug-b'), data.drugs);
  fillSelect($('cell-line'), data.cell_lines);
  $('drug-b').selectedIndex = Math.min(1, data.drugs.length - 1);
  $('model-status').textContent = 'multimodal XGBoost ready';
  document.querySelector('.status-dot').style.background = '#0d685b';
  const validation = data.validation || {};
  $('metric-rmse').textContent = Number(validation.rmse).toFixed(2);
  $('metric-mae').textContent = Number(validation.mae).toFixed(2);
  $('metric-spearman').textContent = Number(validation.spearman).toFixed(2);
  $('metric-rows').textContent = Number(data.data.rows || 0).toLocaleString();
}

$('swap').addEventListener('click', () => {
  const first = $('drug-a').value;
  $('drug-a').value = $('drug-b').value;
  $('drug-b').value = first;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  $('form-error').textContent = '';
  const button = form.querySelector('button[type=submit]');
  button.disabled = true;
  button.firstChild.textContent = 'Calculating ';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({drug_a: $('drug-a').value, drug_b: $('drug-b').value, cell_line: $('cell-line').value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed');
    $('empty-state').hidden = true;
    $('result-content').hidden = false;
    $('result-panel').classList.remove('empty');
    $('score').textContent = data.predicted_bliss.toFixed(2);
    $('label').textContent = data.interpretation;
    $('interval').textContent = `${data.validation_interval_80[0].toFixed(2)} to ${data.validation_interval_80[1].toFixed(2)}`;
    $('cancer-type').textContent = data.cancer_type || 'Unknown';
    $('tissue').textContent = data.tissue || 'Unknown';
    const position = Math.max(0, Math.min(100, ((data.predicted_bliss + 25) / 50) * 100));
    $('range-point').style.left = `calc(${position}% - 8px)`;
  } catch (error) {
    $('form-error').textContent = error.message;
  } finally {
    button.disabled = false;
    button.firstChild.textContent = 'Predict Bliss score ';
  }
});

$('batch-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('batch-error').textContent = '';
  const file = $('batch-file').files[0];
  if (!file) return;
  const body = new FormData(); body.append('file', file);
  try {
    const response = await fetch('/api/batch', {method: 'POST', body});
    if (!response.ok) {
      const data = await response.json(); throw new Error(data.error || 'Batch prediction failed');
    }
    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob); link.download = 'synergy_predictions.csv'; link.click();
    URL.revokeObjectURL(link.href);
  } catch (error) { $('batch-error').textContent = error.message; }
});

loadMetadata().catch(error => {
  $('model-status').textContent = error.message;
  $('form-error').textContent = 'The model must be trained before the interface can make predictions.';
});
