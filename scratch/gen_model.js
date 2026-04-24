import { trainModel, generateSyntheticData } from '../src/ml/trainer.js';
import fs from 'fs';

async function generate() {
    const data = generateSyntheticData(2000);
    const model = await trainModel(data, { nTrees: 8, lr: 0.2, maxDepth: 3 });
    process.stdout.write(JSON.stringify(model));
}

generate();
