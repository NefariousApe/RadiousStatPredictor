import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { loadModels } from './predict.js';
import type { Unit } from './predict.js';

const MODELS_DIR = './onnx_models';
const TEST_FILE = './test.json';

async function main() {
  // Accept [{unit_data: {...}}, ...] or a single {unit_data: {...}}.
  const saved = JSON.parse(await readFile(TEST_FILE, 'utf8'));
  const records: { unit_data: Unit }[] = Array.isArray(saved) ? saved : [saved];
  const units: Unit[] = [];
  for (const record of records) {
    assert.ok(record.unit_data, 'Each test entry needs a unit_data object');
    units.push(record.unit_data);
  }
  assert.ok(units.length > 0, 'Test file contains no units');

  // Loading is outside prediction timing.
  const predictor = await loadModels(MODELS_DIR);
  try {
    const changes = await predictor.predictUnit(records[3].unit_data);
    console.log("Predicted changes:", changes);
} finally {
    await predictor.close();
}
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
