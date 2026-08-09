import Papa from 'papaparse';

export interface ParseCsvResult {
  data: any[];
  errors: Papa.ParseError[];
  meta: Papa.ParseMeta;
}

export function parseCsvContent(fileOrString: File | string): Promise<ParseCsvResult> {
  return new Promise((resolve, reject) => {
    Papa.parse(fileOrString as any, {
      header: true,
      skipEmptyLines: 'greedy',
      dynamicTyping: false, // Keep raw strings to detect NA/NaN/Inf manually
      complete: (results) => {
        resolve({
          data: results.data,
          errors: results.errors,
          meta: results.meta,
        });
      },
      error: (error) => {
        reject(error);
      },
    });
  });
}

/**
 * Converts array of prediction objects to CSV string download
 */
export function convertToCsv(data: Record<string, any>[]): string {
  return Papa.unparse(data);
}
