import time
from src.data_loader import FootballDataLoader
from src.preprocessing import FootballPreprocessor
from src.feature_engineering import FeatureExtractor
from src.model_training import ModelTrainer
from src.simulation import TournamentSimulator
from src.utils import setup_logger

logger = setup_logger("pipeline_runner")

def run_full_pipeline():
    logger.info("=============================================")
    logger.info("🏆 STARTING FOOTBALL PREDICTOR ML PIPELINE 🏆")
    logger.info("=============================================")
    start_time = time.time()

    # Step 1: Ingest & Preprocess
    logger.info("[STEP 1/4] Running Data Ingestion & Preprocessing...")
    s1_start = time.time()
    try:
        loader = FootballDataLoader()
        r, s, rk, c = loader.load_raw_data()
        preprocessor = FootballPreprocessor()
        preprocessor.process_and_save(r, s, rk, c)
        logger.info(f"Step 1 Complete. Time elapsed: {time.time() - s1_start:.2f}s\n")
    except Exception as e:
        logger.error(f"Step 1 Failed: {e}")
        return

    # Step 2: Feature Extraction
    logger.info("[STEP 2/4] Running Feature Extraction...")
    s2_start = time.time()
    try:
        extractor = FeatureExtractor()
        extractor.extract_all_features()
        logger.info(f"Step 2 Complete. Time elapsed: {time.time() - s2_start:.2f}s\n")
    except Exception as e:
        logger.error(f"Step 2 Failed: {e}")
        return

    # Step 3: Model Training & Calibration
    logger.info("[STEP 3/4] Running Model Training & Calibration...")
    s3_start = time.time()
    try:
        trainer = ModelTrainer()
        trainer.train_and_compare()
        logger.info(f"Step 3 Complete. Time elapsed: {time.time() - s3_start:.2f}s\n")
    except Exception as e:
        logger.error(f"Step 3 Failed: {e}")
        return

    # Step 4: Verification Simulation
    logger.info("[STEP 4/4] Running Verification Simulation...")
    s4_start = time.time()
    try:
        simulator = TournamentSimulator()
        # Run a quick 500-sim verification
        simulator.n_simulations = 500
        results = simulator.run_monte_carlo()
        
        logger.info("Top Contenders from verification run:")
        for team, prob in results["champions_probs"][:3]:
            logger.info(f" - {team}: {prob*100:.1f}%")
        logger.info(f"Step 4 Complete. Time elapsed: {time.time() - s4_start:.2f}s\n")
    except Exception as e:
        logger.error(f"Step 4 Failed: {e}")
        return

    total_time = time.time() - start_time
    logger.info("=============================================")
    logger.info(f"🎉 PIPELINE RUN COMPLETED SUCCESSFUL IN {total_time:.2f}s 🎉")
    logger.info("=============================================")

    # [DEBUG print execution details]
    # print(f"[DEBUG PIPELINE] start={start_time} end={time.time()} elapsed={total_time:.2f}s")

if __name__ == "__main__":
    run_full_pipeline()
