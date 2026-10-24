import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import * as ort from 'onnxruntime-node';

export type Unit = Record<string, string | number | boolean | null | undefined>;
export type StatChanges = Record<string, number>;

type ModelEntry = { file: string } | { constant: number };
type PreprocessorSchema = {
  inputs: { name: string; type: 'float32' | 'string' }[];
  categorical_missing: string;
  output: string;
};

/** Call once, reuse the returned predictor, then close it when unloading the mod. */
export async function loadModels(directory = './onnx_models') {
  const manifest: Record<string, ModelEntry> = JSON.parse(
    await readFile(join(directory, 'models.json'), 'utf8')
  );
  const schema: PreprocessorSchema = JSON.parse(
    await readFile(join(directory, 'preprocessor.json'), 'utf8')
  );

  // These models run sequentially; avoid a separate large CPU thread pool per model.
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['cpu'],
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  };
  const preprocessor = await ort.InferenceSession.create(
    join(directory, 'preprocessor.onnx'), options
  );
  const models = new Map<string, ort.InferenceSession | number>();
  try {
    for (const [target, entry] of Object.entries(manifest)) {
      if ('constant' in entry) {
        models.set(target, entry.constant);
      } else {
        models.set(target, await ort.InferenceSession.create(join(directory, entry.file), options));
      }
    }
  } catch (error) {
    await preprocessor.release();
    for (const model of models.values()) {
      if (typeof model !== 'number') await model.release();
    }
    throw error;
  }

  async function predictUnits(units: Unit[]): Promise<StatChanges[]> {
    if (units.length === 0) return [];
    const feeds: Record<string, ort.Tensor> = {};

    // Select only schema inputs. Unit keys and the other join references are ignored.
    for (const input of schema.inputs) {
      const numbers = new Float32Array(units.length);
      const strings: string[] = [];
      for (let row = 0; row < units.length; row++) {
        const unit = units[row];
        let value = unit[input.name];

        // This feature was derived in Python, so it isn't in the cleaned JSON.
        if (input.name === 'is_ranged') {
          if (!('land_units__primary_missile_weapon' in unit)) {
            throw new Error('Missing land_units__primary_missile_weapon for is_ranged');
          }
          value = unit.land_units__primary_missile_weapon == null ? 0 : 1;
        } else if (!(input.name in unit)) {
          throw new Error(`Missing expected input field: ${input.name}`);
        }

        if (input.type === 'string') {
          if (value == null) strings.push(schema.categorical_missing);
          else if (typeof value === 'string') strings.push(value);
          else throw new Error(`Expected string or null: ${input.name}`);
        } else {
          // Leave numeric missingness as NaN; preprocessor.onnx performs zero-fill.
          if (value == null) numbers[row] = NaN;
          else if (typeof value === 'boolean') numbers[row] = Number(value);
          else if (typeof value === 'number' && Number.isFinite(value)) numbers[row] = value;
          else throw new Error(`Expected finite number or null: ${input.name}`);
        }
      }
      feeds[input.name] = input.type === 'string'
        ? new ort.Tensor('string', strings, [units.length, 1])
        : new ort.Tensor('float32', numbers, [units.length, 1]);
    }

    const preprocessed = await preprocessor.run(feeds);
    const features = preprocessed[schema.output];
    if (!features || features.type !== 'float32' || features.dims[0] !== units.length) {
      throw new Error('Unexpected preprocessor output; expected float32 [batch, features]');
    }

    const changes: StatChanges[] = [];
    for (let row = 0; row < units.length; row++) changes.push({});

    for (const [target, model] of models) {
      const stat = target.replace(/^delta_/, '');
      // skip when unit doesnt have projectile
      if (stat.startsWith("projectiles__")) {
            const hasProjectile = units.some(
                unit => unit["projectiles__key"] != null
            );

            if (!hasProjectile) {
                for (let row = 0; row < units.length; row++) {
                    changes[row][stat] = 0;
                }

                continue;
            }
        }
      if (typeof model === 'number') {
        for (let row = 0; row < units.length; row++) changes[row][stat] = model;
      } else {
        const output = await model.run({ [model.inputNames[0]]: features });
        const predictions = output[model.outputNames[0]];
        if (predictions.data.length !== units.length) {
          throw new Error(`Expected one prediction per unit: ${target}`);
        }
        for (let row = 0; row < units.length; row++) {
          const value = Number(predictions.data[row]);
          if (!Number.isFinite(value)) throw new Error(`Nonfinite prediction: ${target}`);
          const rounded = Math.round(value);
          changes[row][stat] = rounded;
        }
      }
    }

    return changes;
  }

  async function predictUnit(unit: Unit): Promise<StatChanges> {
    const changes = await predictUnits([unit]);
    return changes[0];
  }

  async function close() {
    await preprocessor.release();
    for (const model of models.values()) {
      if (typeof model !== 'number') await model.release();
    }
  }

  return { predictUnit, predictUnits, close };
}
