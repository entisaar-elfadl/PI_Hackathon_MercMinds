/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GroundTruth, Prediction } from '../types';

/**
 * A seedable deterministic pseudo-random number generator (LCG)
 * so we generate the exact same datasets every time on the client.
 */
class SeededRandom {
  private m = 0x80000000; // 2**31
  private a = 1103515245;
  private c = 12345;
  private state: number;

  constructor(seed: number) {
    this.state = seed ? seed : Math.floor(Math.random() * (this.m - 1));
  }

  // Returns number between 0 and 1
  next(): number {
    this.state = (this.a * this.state + this.c) % this.m;
    return this.state / (this.m - 1);
  }

  // Returns integer in range [min, max]
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  // Choose from array
  choice<T>(arr: T[]): T {
    const idx = this.nextInt(0, arr.length - 1);
    return arr[idx];
  }
}

/**
 * Generate a realistic survey response for a participant
 */
function generateParticipant(rng: SeededRandom, id: string, roundNum: number): GroundTruth {
  const age = rng.nextInt(18, 45);
  const gender = rng.choice(['Female', 'Male', 'Female', 'Male', 'Other']);
  
  const province = rng.choice([
    'Gauteng', 'Gauteng', 'Western Cape', 'Western Cape', 
    'KwaZulu-Natal', 'Eastern Cape', 'Limpopo', 'Mpumalanga', 
    'Free State', 'North West', 'Northern Cape'
  ]);

  const education = rng.choice([
    'None', 'Primary', 'Grade 10-11', 
    'Matric (Grade 12)', 'Matric (Grade 12)', 
    'Tertiary Certificate', 'University Degree'
  ]);

  // Soft-skills and job aptitude assessment score
  let workReadiness = rng.nextInt(15, 85);
  // Education improves work readiness score
  if (education === 'University Degree') workReadiness += rng.nextInt(10, 15);
  else if (education === 'Tertiary Certificate') workReadiness += rng.nextInt(5, 10);
  else if (education === 'None') workReadiness -= rng.nextInt(5, 15);
  workReadiness = Math.max(0, Math.min(100, workReadiness));

  // Unemployment duration in years
  let yearsUnemployed = rng.nextInt(0, 8);
  if (age < 22) yearsUnemployed = Math.min(yearsUnemployed, age - 18);

  const hasInternet = rng.next() > 0.3 ? 'Yes' : 'No';
  const householdSize = rng.nextInt(1, 8);

  // Math model determining probability of employment status:
  // p = sigmoid(beta_0 + beta_1 * readiness + beta_2 * years_unemployed + beta_3 * degree + beta_4 * matric + noise)
  let logOdds = -2.2; // intercept
  logOdds += 0.045 * workReadiness;
  logOdds -= 0.28 * yearsUnemployed;
  if (education === 'University Degree') logOdds += 1.8;
  else if (education === 'Tertiary Certificate') logOdds += 1.0;
  else if (education === 'Matric (Grade 12)') logOdds += 0.5;
  if (hasInternet === 'Yes') logOdds += 0.4;
  logOdds -= 0.05 * Math.abs(age - 28); // optimal employment age centered around 28
  
  // Add some randomness
  logOdds += (rng.next() - 0.5) * 1.5;

  const probability = 1 / (1 + Math.exp(-logOdds));
  const employedStatus = rng.next() < probability ? 1 : 0;

  return {
    anonymised_id: id,
    survey_round: roundNum,
    age,
    gender,
    province,
    education_level: education,
    work_readiness_score: workReadiness,
    years_unemployed: yearsUnemployed,
    has_internet_access: hasInternet,
    household_size: householdSize,
    employed_status: employedStatus,
  };
}

/**
 * Generate datasets for Rounds 6, 7, and 8 deterministically
 */
export function generateRoundDatasets(roundId: 'round_6' | 'round_7' | 'round_8'): {
  training: GroundTruth[];
  evaluation: GroundTruth[];
} {
  const training: GroundTruth[] = [];
  const evaluation: GroundTruth[] = [];

  // Seed ranges to guarantee uniqueness and consistency
  // Round 6: Train = Rounds 1-5, Eval = Round 6
  // Round 7: Train = Rounds 1-6, Eval = Round 7
  // Round 8: Train = Rounds 1-7, Eval = Round 8

  if (roundId === 'round_6') {
    // Rounds 1-5 (Training)
    const rngTrain = new SeededRandom(1005);
    for (let i = 1; i <= 3800; i++) {
      const roundNum = rngTrain.nextInt(1, 5);
      training.push(generateParticipant(rngTrain, `ID_TR6_${i}`, roundNum));
    }
    // Round 6 (Evaluation)
    const rngEval = new SeededRandom(2006);
    for (let i = 1; i <= 1200; i++) {
      evaluation.push(generateParticipant(rngEval, `ID_R6_${i}`, 6));
    }
  } else if (roundId === 'round_7') {
    // Rounds 1-6 (Training)
    const rngTrain = new SeededRandom(1006);
    for (let i = 1; i <= 4600; i++) {
      const roundNum = rngTrain.nextInt(1, 6);
      training.push(generateParticipant(rngTrain, `ID_TR7_${i}`, roundNum));
    }
    // Round 7 (Evaluation)
    const rngEval = new SeededRandom(2007);
    for (let i = 1; i <= 1300; i++) {
      evaluation.push(generateParticipant(rngEval, `ID_R7_${i}`, 7));
    }
  } else if (roundId === 'round_8') {
    // Rounds 1-7 (Training)
    const rngTrain = new SeededRandom(1007);
    for (let i = 1; i <= 5400; i++) {
      const roundNum = rngTrain.nextInt(1, 7);
      training.push(generateParticipant(rngTrain, `ID_TR8_${i}`, roundNum));
    }
    // Round 8 (Evaluation)
    const rngEval = new SeededRandom(2008);
    for (let i = 1; i <= 1400; i++) {
      evaluation.push(generateParticipant(rngEval, `ID_R8_${i}`, 8));
    }
  }

  return { training, evaluation };
}

/**
 * Generate a prediction file from ground truths representing model tiers.
 */
export function generateSamplePredictions(
  evaluationData: GroundTruth[],
  modelType: 'perfect' | 'random' | 'weak_logistic' | 'strong_boosting',
  seedValue = 42
): Prediction[] {
  const rng = new SeededRandom(seedValue);
  
  return evaluationData.map(p => {
    let score = 0.5;

    // Retrieve underlying attributes to calculate realistic probabilities
    const age = Number(p.age) || 28;
    const workReadiness = Number(p.work_readiness_score) || 50;
    const yearsUnemployed = Number(p.years_unemployed) || 2;
    const isDegree = p.education_level === 'University Degree';
    const isMatric = p.education_level === 'Matric (Grade 12)';
    const actual = p.employed_status;

    if (modelType === 'perfect') {
      // 100% correct ranking
      score = actual === 1 ? rng.nextInt(85, 100) / 100 : rng.nextInt(0, 15) / 100;
    } else if (modelType === 'random') {
      // Complete noise
      score = rng.next();
    } else if (modelType === 'weak_logistic') {
      // Uses only work_readiness_score and age (weak linear combination)
      let logOdds = -1.0 + 0.02 * workReadiness - 0.03 * Math.abs(age - 28);
      // Small random perturbation
      logOdds += (rng.next() - 0.5) * 1.5;
      score = 1 / (1 + Math.exp(-logOdds));
    } else if (modelType === 'strong_boosting') {
      // Captures strong underlying features (Degree, Readiness, Unemployed, etc.)
      let logOdds = -1.8;
      logOdds += 0.038 * workReadiness;
      logOdds -= 0.22 * yearsUnemployed;
      if (isDegree) logOdds += 1.4;
      if (isMatric) logOdds += 0.4;
      // High-quality fit
      logOdds += (actual === 1 ? 0.8 : -0.8);
      // Slight noise
      logOdds += (rng.next() - 0.5) * 0.5;
      
      score = 1 / (1 + Math.exp(-logOdds));
    }

    // Keep strict 5 decimal places for clean R/Kaggle likeness
    score = parseFloat(score.toFixed(5));

    return {
      anonymised_id: p.anonymised_id,
      employed_status: score,
    };
  });
}
