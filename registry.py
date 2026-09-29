# ============================================================
# Checkpoint Registry
# ============================================================
# Checkpoint paths used for evaluation, one dictionary per training
# regime. Keys are BirdSet task names (DT) or "ALL" (MT, LT, LT_SSA:
# one set of checkpoints used for every task).
#
# Structure:
#   {
#       "<TASK_NAME>": [
#           "<checkpoint_path_1>",
#           "<checkpoint_path_2>",
#           "<checkpoint_path_3>",
#       ]
#   }
#
# Notes:
# - Each entry holds 3 independently trained runs.
# - Evaluation reports their mean.
# ============================================================


# ------------------------------------------------------------
# Downstream Task (DT) checkpoints
# ------------------------------------------------------------
# Each key corresponds to a specific downstream dataset.
# Used when evaluating task-specific fine-tuned models.

DT = {
    "HSN": ["ckpts/DT/HSN/dsa_HSN_seed1",
            "ckpts/DT/HSN/dsa_HSN_seed2",
            "ckpts/DT/HSN/dsa_HSN_seed3"],

    "POW": ["ckpts/DT/POW/dsa_POW_seed1",
            "ckpts/DT/POW/dsa_POW_seed2",
            "ckpts/DT/POW/dsa_POW_seed3"],

    "SNE": ["ckpts/DT/SNE/dsa_SNE_seed1",
            "ckpts/DT/SNE/dsa_SNE_seed2",
            "ckpts/DT/SNE/dsa_SNE_seed3"],

    "PER": ["ckpts/DT/PER/dsa_PER_seed1",
            "ckpts/DT/PER/dsa_PER_seed2",
            "ckpts/DT/PER/dsa_PER_seed3"],

    "NES": ["ckpts/DT/NES/dsa_NES_seed1",
            "ckpts/DT/NES/dsa_NES_seed2",
            "ckpts/DT/NES/dsa_NES_seed3"],

    "UHH": ["ckpts/DT/UHH/dsa_UHH_seed1",
            "ckpts/DT/UHH/dsa_UHH_seed2",
            "ckpts/DT/UHH/dsa_UHH_seed3"],

    "NBP": ["ckpts/DT/NBP/dsa_NBP_seed1",
            "ckpts/DT/NBP/dsa_NBP_seed2",
            "ckpts/DT/NBP/dsa_NBP_seed3"],

    "SSW": ["ckpts/DT/SSW/dsa_SSW_seed1",
            "ckpts/DT/SSW/dsa_SSW_seed2",
            "ckpts/DT/SSW/dsa_SSW_seed3"],
}

MT = {"ALL": ["ckpts/MT/dsa_MT_seed1",
              "ckpts/MT/dsa_MT_seed2",
              "ckpts/MT/dsa_MT_seed3"]
      }

# Knowledge-distilled DSA runs, one per seed, over XCL's
# 9,736-class bird label space.
LT = {"ALL": ["ckpts/LT/dsa_LT_seed1",
              "ckpts/LT/dsa_LT_seed2",
              "ckpts/LT/dsa_LT_seed3"]
      }

# The same three LT runs with the single-branch SSA head: each is the
# fine-grained branch of the DSA model with the same seed, not a separate
# training run.
LT_SSA = {"ALL": ["ckpts/LT/ssa_LT_seed1",
                  "ckpts/LT/ssa_LT_seed2",
                  "ckpts/LT/ssa_LT_seed3"]
          }
