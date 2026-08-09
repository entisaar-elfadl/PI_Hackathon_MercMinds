import { GroundTruthRow, RoundId, RoundInfo } from '../types';

export const HISTORICAL_ROUNDS: Record<RoundId, RoundInfo> = {
  round_6: {
    id: 'round_6',
    title: 'Round 6 Historical Evaluation',
    trainingSet: 'Rounds 1–5 (rounds_1_5.csv)',
    evaluationSet: 'Round 6 (round_6.csv)',
    description: 'Simulates evaluating model performance on Round 6 survey data after training strictly on Rounds 1–5.',
    sampleObservationCount: 1000,
    groundTruthFile: 'round_6.csv',
    trainingDataFile: 'rounds_1_5.csv',
  },
  round_7: {
    id: 'round_7',
    title: 'Round 7 Historical Evaluation',
    trainingSet: 'Rounds 1–6 (rounds_1_6.csv)',
    evaluationSet: 'Round 7 (round_7.csv)',
    description: 'Simulates evaluating model performance on Round 7 survey data after training strictly on Rounds 1–6.',
    sampleObservationCount: 1150,
    groundTruthFile: 'round_7.csv',
    trainingDataFile: 'rounds_1_6.csv',
  },
  round_8: {
    id: 'round_8',
    title: 'Round 8 Historical Evaluation',
    trainingSet: 'Rounds 1–7 (rounds_1_7.csv)',
    evaluationSet: 'Round 8 (round_8.csv)',
    description: 'Simulates evaluating model performance on Round 8 survey data after training strictly on Rounds 1–7.',
    sampleObservationCount: 1300,
    groundTruthFile: 'round_8.csv',
    trainingDataFile: 'rounds_1_7.csv',
  },
};

/**
 * Deterministic synthetic ground truth generation for realistic simulation
 */
function generateSyntheticGroundTruth(roundId: RoundId, count: number): GroundTruthRow[] {
  const rows: GroundTruthRow[] = [];
  // Seed based on roundId
  const seed = roundId === 'round_6' ? 600 : roundId === 'round_7' ? 700 : 800;

  for (let i = 1; i <= count; i++) {
    const idNum = seed * 1000 + i;
    const id = `ID_${idNum}`;

    // Synthetic feature generation
    const age = 18 + Math.floor(((i * 17 + seed) % 45));
    const eduLevel = Math.floor((i * 3 + seed) % 5) + 1; // 1 to 5
    const readiness = Math.round((((i * 13 + seed * 3) % 100) / 100) * 10) / 10; // 0.0 to 1.0
    const prevMonths = Math.floor((i * 19 + seed * 7) % 60);

    // Logistic formula for true employment status probability
    const logit = -1.5 + age * 0.02 + eduLevel * 0.35 + readiness * 1.8 + prevMonths * 0.03 - (roundId === 'round_8' ? 0.2 : 0);
    const trueProb = 1 / (1 + Math.exp(-logit));

    // Determine binary ground truth (0 or 1) based on threshold
    const rand = ((i * 31 + seed * 11) % 100) / 100;
    const status = rand < trueProb ? 1 : 0;

    rows.push({
      anonymised_id: id,
      employed_status: status,
      age,
      education_level: eduLevel,
      work_readiness_score: readiness,
      prev_employment_months: prevMonths,
      region: `Region_${(i % 6) + 1}`,
    });
  }

  return rows;
}

// Pre-generated historical ground truth datasets for client-side execution
export const HISTORICAL_GROUND_TRUTH: Record<RoundId, GroundTruthRow[]> = {
  round_6: generateSyntheticGroundTruth('round_6', 1000),
  round_7: generateSyntheticGroundTruth('round_7', 1150),
  round_8: generateSyntheticGroundTruth('round_8', 1300),
};
