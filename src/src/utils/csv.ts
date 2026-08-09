/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Prediction, GroundTruth, ValidationError } from '../types';

/**
 * A robust, lightweight CSV parser that handles double-quoted values,
 * escaped characters, commas, and CRLF/LF line endings.
 */
export function parseCSV(text: string): string[][] {
  const result: string[][] = [];
  let row: string[] = [];
  let inQuotes = false;
  let currentValue = '';

  const cleanText = text.trim();
  if (!cleanText) return [];

  for (let i = 0; i < cleanText.length; i++) {
    const char = cleanText[i];
    const nextChar = cleanText[i + 1];

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        // Escaped quote
        currentValue += '"';
        i++; // skip next quote
      } else {
        // Toggle quote state
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      row.push(currentValue.trim());
      currentValue = '';
    } else if ((char === '\r' || char === '\n') && !inQuotes) {
      row.push(currentValue.trim());
      result.push(row);
      row = [];
      currentValue = '';
      if (char === '\r' && nextChar === '\n') {
        i++; // skip LF after CR
      }
    } else {
      currentValue += char;
    }
  }

  // Push final value and row if needed
  if (currentValue || row.length > 0) {
    row.push(currentValue.trim());
    result.push(row);
  }

  // Filter out any completely empty rows
  return result.filter(r => r.length > 0 && r.some(cell => cell !== ''));
}

/**
 * Validate prediction CSV data according to the competition rules.
 */
export function validatePredictionCSV(
  rows: string[][]
): { predictions: Prediction[]; errors: ValidationError[] } {
  const predictions: Prediction[] = [];
  const errors: ValidationError[] = [];

  if (rows.length === 0) {
    errors.push({
      message: 'The prediction file is empty.',
      severity: 'error',
    });
    return { predictions, errors };
  }

  const headers = rows[0].map(h => h.toLowerCase());
  const idColIdx = headers.indexOf('anonymised_id');
  const statusColIdx = headers.indexOf('employed_status');

  if (idColIdx === -1) {
    errors.push({
      message: 'Missing required column: "anonymised_id" (case-insensitive).',
      severity: 'error',
    });
  }
  if (statusColIdx === -1) {
    errors.push({
      message: 'Missing required column: "employed_status" (case-insensitive).',
      severity: 'error',
    });
  }

  if (errors.length > 0) {
    return { predictions, errors };
  }

  const seenIds = new Set<string>();

  for (let i = 1; i < rows.length; i++) {
    const row = rows[i];
    const rowNum = i + 1;

    // Handle mismatched columns length
    if (row.length <= Math.max(idColIdx, statusColIdx)) {
      errors.push({
        row: rowNum,
        message: `Row ${rowNum} is truncated or has missing columns.`,
        severity: 'error',
      });
      continue;
    }

    const rawId = row[idColIdx];
    const rawStatus = row[statusColIdx];

    if (!rawId) {
      errors.push({
        row: rowNum,
        column: 'anonymised_id',
        message: `Row ${rowNum}: "anonymised_id" is empty.`,
        severity: 'error',
      });
      continue;
    }

    if (seenIds.has(rawId)) {
      errors.push({
        row: rowNum,
        column: 'anonymised_id',
        value: rawId,
        message: `Row ${rowNum}: Duplicate ID "${rawId}" found. Every participant must have exactly one prediction.`,
        severity: 'error',
      });
      continue;
    }

    seenIds.add(rawId);

    if (rawStatus === '' || rawStatus === undefined) {
      errors.push({
        row: rowNum,
        column: 'employed_status',
        message: `Row ${rowNum}: Predicted probability is empty.`,
        severity: 'error',
      });
      continue;
    }

    const prob = Number(rawStatus);

    if (isNaN(prob)) {
      errors.push({
        row: rowNum,
        column: 'employed_status',
        value: rawStatus,
        message: `Row ${rowNum}: Predicted value "${rawStatus}" is not a valid number (e.g. NA, NaN, Inf).`,
        severity: 'error',
      });
      continue;
    }

    if (!isFinite(prob)) {
      errors.push({
        row: rowNum,
        column: 'employed_status',
        value: rawStatus,
        message: `Row ${rowNum}: Predicted value is infinite (${rawStatus}).`,
        severity: 'error',
      });
      continue;
    }

    if (prob < 0 || prob > 1) {
      errors.push({
        row: rowNum,
        column: 'employed_status',
        value: rawStatus,
        message: `Row ${rowNum}: Probability ${prob} is outside the valid range [0, 1]. Predictions must be probabilities.`,
        severity: 'error',
      });
      continue;
    }

    predictions.push({
      anonymised_id: rawId,
      employed_status: prob,
    });
  }

  return { predictions, errors };
}

/**
 * Validate ground truth CSV file.
 */
export function validateGroundTruthCSV(
  rows: string[][]
): { groundTruths: GroundTruth[]; errors: ValidationError[] } {
  const groundTruths: GroundTruth[] = [];
  const errors: ValidationError[] = [];

  if (rows.length === 0) {
    errors.push({
      message: 'The ground truth file is empty.',
      severity: 'error',
    });
    return { groundTruths, errors };
  }

  const headers = rows[0].map(h => h.toLowerCase());
  const idColIdx = headers.indexOf('anonymised_id');
  const statusColIdx = headers.indexOf('employed_status');

  if (idColIdx === -1) {
    errors.push({
      message: 'Ground truth missing required column: "anonymised_id".',
      severity: 'error',
    });
  }
  if (statusColIdx === -1) {
    errors.push({
      message: 'Ground truth missing required column: "employed_status".',
      severity: 'error',
    });
  }

  if (errors.length > 0) {
    return { groundTruths, errors };
  }

  const seenIds = new Set<string>();

  for (let i = 1; i < rows.length; i++) {
    const row = rows[i];
    const rowNum = i + 1;

    if (row.length <= Math.max(idColIdx, statusColIdx)) {
      errors.push({
        row: rowNum,
        message: `Row ${rowNum} in ground truth is truncated.`,
        severity: 'warning',
      });
      continue;
    }

    const rawId = row[idColIdx];
    const rawStatus = row[statusColIdx];

    if (!rawId) {
      errors.push({
        row: rowNum,
        column: 'anonymised_id',
        message: `Row ${rowNum}: "anonymised_id" is empty.`,
        severity: 'warning',
      });
      continue;
    }

    if (seenIds.has(rawId)) {
      errors.push({
        row: rowNum,
        column: 'anonymised_id',
        value: rawId,
        message: `Row ${rowNum}: Duplicate ID "${rawId}" in ground truth data.`,
        severity: 'warning',
      });
      continue;
    }

    seenIds.add(rawId);

    const status = Number(rawStatus);

    if (rawStatus === '' || rawStatus === undefined || isNaN(status)) {
      errors.push({
        row: rowNum,
        column: 'employed_status',
        message: `Row ${rowNum}: Target status is missing or not a number.`,
        severity: 'warning',
      });
      continue;
    }

    if (status !== 0 && status !== 1) {
      errors.push({
        row: rowNum,
        column: 'employed_status',
        value: rawStatus,
        message: `Row ${rowNum}: Target status must be strictly binary: 0 (unemployed) or 1 (employed). Found: ${status}`,
        severity: 'warning',
      });
      continue;
    }

    // Capture extra columns
    const record: GroundTruth = {
      anonymised_id: rawId,
      employed_status: status,
    };

    // Attach any extra data dynamically
    headers.forEach((header, index) => {
      if (index !== idColIdx && index !== statusColIdx && index < row.length) {
        record[header] = row[index];
      }
    });

    groundTruths.push(record);
  }

  return { groundTruths, errors };
}

/**
 * Helper to convert array of objects to CSV download string
 */
export function convertToCSV(data: any[]): string {
  if (data.length === 0) return '';
  const headers = Object.keys(data[0]);
  const csvRows = [headers.join(',')];

  for (const row of data) {
    const values = headers.map(header => {
      const val = row[header];
      if (val === null || val === undefined) return '';
      const stringVal = String(val);
      if (stringVal.includes(',') || stringVal.includes('"') || stringVal.includes('\n')) {
        return `"${stringVal.replace(/"/g, '""')}"`;
      }
      return stringVal;
    });
    csvRows.push(values.join(','));
  }

  return csvRows.join('\n');
}
