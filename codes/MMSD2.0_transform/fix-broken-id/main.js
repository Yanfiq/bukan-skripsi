const fs = require('fs');
const path = require('path');

const inputPrePath = path.join(__dirname, 'dataset_pre.json');
const inputTranslatedPath = path.join(__dirname, 'dataset_translated.json');
const outputPath = path.join(__dirname, 'dataset_translated_fixed.json');

function preserveLargeImageIds(jsonText) {
	return jsonText.replace(/("image_id"\s*:\s*)(\d+)/g, '$1"$2"');
}

function restoreLargeImageIds(jsonText) {
	return jsonText.replace(/("image_id"\s*:\s*)"(\d+)"/g, '$1$2');
}

function loadDataset(filePath) {
	const rawText = fs.readFileSync(filePath, 'utf8');
	return JSON.parse(preserveLargeImageIds(rawText));
}

function sameRowExceptTranslation(preItem, translatedItem) {
	return (
		preItem.image_id === translatedItem.image_id &&
		preItem.text === translatedItem.text &&
		preItem.label === translatedItem.label &&
		preItem.split === translatedItem.split
	);
}

function fixDataset() {
	const preData = loadDataset(inputPrePath);
	const translatedData = loadDataset(inputTranslatedPath);
	const mismatchSamples = [];
	let mismatchCount = 0;

	if (preData.length !== translatedData.length) {
		throw new Error(`Length mismatch: pre=${preData.length}, translated=${translatedData.length}`);
	}

	const fixedData = translatedData.map((translatedItem, index) => {
		const preItem = preData[index];

		if (!preItem) {
			return translatedItem;
		}

		if (!sameRowExceptTranslation(preItem, translatedItem)) {
			mismatchCount += 1;
			if (mismatchSamples.length < 5) {
				mismatchSamples.push({
					index: index + 1,
					from: translatedItem.image_id,
					to: preItem.image_id,
				});
			}
		}

		return {
			...preItem,
			text_translated: translatedItem.text_translated,
		};
	});

	const outputText = restoreLargeImageIds(JSON.stringify(fixedData, null, 2)) + '\n';
	fs.writeFileSync(outputPath, outputText, 'utf8');

	console.log(`Wrote corrected dataset to ${outputPath}`);

	if (mismatchCount > 0) {
		console.log(`Repaired ${mismatchCount} row(s) whose metadata did not match dataset_pre.json.`);
		console.log('Sample mismatches:', mismatchSamples);
	} else {
		console.log('All rows already matched dataset_pre.json except text_translated.');
	}
}

fixDataset();
