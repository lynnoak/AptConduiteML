import pandas as pd
from pathlib import Path

# ===== Path configuration =====

FILE_PATH = Path("V3_APTICONDUITE_Excel_datas_hors_simu.xlsx")
SAVE_PATH = Path(".\output")


def normalize_text(value):
    """Normalize text values for robust string matching."""
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    if value in ["nan", "none", "na", "n/a"]:
        return ""
    return value


def parse_kms_level(kms_value):
    """Convert kms text into an ordinal annual mileage level."""
    value = normalize_text(kms_value)

    if value in ["", "0", "non"]:
        return 0
    if "entre 1 et 1000" in value:
        return 1
    if "entre 1001 et 10000" in value:
        return 2
    if "entre 10001 et 50000" in value:
        return 3
    if ">50000" in value or "plus de 50000" in value:
        return 4

    return 0


def parse_frequency_level(freq_value):
    """Convert driving frequency text into an ordinal level."""
    value = normalize_text(freq_value)

    mapping = {
        "jamais": 0,
        "occasionnellement": 1,
        "regulierement": 2,
        "frequemment": 3,
        "quotidiennement": 4,
        "": 0,
        "0": 0
    }
    return mapping.get(value, 0)


class DataPre:
    handness_tendency_map = {
        "droite": 2,
        "droit": 2,
        "gauche": -2,
        "ambidextre": 0,
        "nc": 0,
        "0": 0,
        "tendance_droite": 1,
        "tendance_gauche": -1,
        -2: -2,
        -1: -1,
        0: 0,
        1: 1,
        2: 2
    }

    def __init__(self, file_path, save_path):
        self.file_path = Path(file_path)
        self.save_path = Path(save_path)

    def clean_input_dataframe(self, df):
        """
        Keep only valid questionnaire rows from Datas-IA-Jiajun.
        Remove empty rows, invalid ids, and duplicated ids.
        """
        df = df.copy()

        # Remove fully empty rows
        df = df.dropna(how="all")

        # Strip column names
        df.columns = df.columns.str.strip()

        # Normalize id after removing full-empty rows
        df["id_anonymat"] = df["id_anonymat"].apply(
            lambda x: "" if pd.isna(x) or str(x).strip() == "0" else str(x).strip()
        )

        # Remove invalid ids
        invalid_ids = ["", "nan", "none", "0", "na", "n/a"]
        df = df[~df["id_anonymat"].str.lower().isin(invalid_ids)].copy()

        # Keep rows with at least one meaningful questionnaire field
        key_cols = [
            "interfacesdeconduite",
            "lunettes",
            "maindom",
            "piedom",
            "oeildom",
            "permis",
            "kms",
            "frequence",
            "boitevitesse"
        ]

        existing_key_cols = [col for col in key_cols if col in df.columns]

        def has_meaningful_data(row):
            for col in existing_key_cols:
                value = normalize_text(row.get(col, ""))
                if value not in ["", "0"]:
                    return True
            return False

        df = df[df.apply(has_meaningful_data, axis=1)].copy()

        # Remove duplicated ids
        df = df.drop_duplicates(subset=["id_anonymat"], keep="first").copy()

        return df

    def ensure_unique_id(self, df, name="dataframe"):
        """Ensure one row per id before merge."""
        df = df.copy()
        df["id_anonymat"] = df["id_anonymat"].apply(
            lambda x: "" if pd.isna(x) or str(x).strip() == "0" else str(x).strip()
        )
        df = df[df["id_anonymat"] != ""].copy()
        df = df.drop_duplicates(subset=["id_anonymat"], keep="first").copy()
        return df

    def classify_interface(self, df, col="interfacesdeconduite"):
        """Encode driving interface type."""
        mapping = {
            "commandes simples": 1,
            "commandes volant": 2,
            "commandes joystick": 3
        }

        result = df[["id_anonymat", col]].copy()
        result["interface"] = result[col].apply(
            lambda x: mapping.get(normalize_text(x), 0)
        )
        return result[["id_anonymat", "interface"]]

    def classify_operational_handness_tendency(self, df, cols=["maindom", "piedom", "oeildom"]):
        """Aggregate hand/foot/eye dominance into a handness_tendency score."""
        subset = df[["id_anonymat"] + cols].copy()

        for col in cols:
            subset[col] = subset[col].apply(
                lambda x: self.handness_tendency_map.get(normalize_text(x), 0)
            )

        subset["handness_tendency_score"] = subset[cols].mean(axis=1)

        def categorize(score):
            if score <= -1.5:
                return -2
            elif score <= -0.5:
                return -1
            elif score < 0.5:
                return 0
            elif score < 1.5:
                return 1
            else:
                return 2

        subset["handness_tendency"] = subset["handness_tendency_score"].apply(categorize)
        return subset[["id_anonymat", "handness_tendency"]]

    def classify_glasses(self, df, col="lunettes"):
        """Encode glasses status."""
        mapping = {
            "oui": -1,
            "non": 1,
            "0": 0,
            "": 0
        }

        result = df[["id_anonymat", col]].copy()
        result["glasses_status"] = result[col].apply(
            lambda x: mapping.get(normalize_text(x), 0)
        )
        return result[["id_anonymat", "glasses_status"]]

    def classify_license_status(self, df, col="permis"):
        """Encode whether the participant has a driving license."""
        mapping = {
            "oui": 1,
            "non": 0,
            "0": 0,
            "": 0
        }

        result = df[["id_anonymat", col]].copy()
        result["license_status"] = result[col].apply(
            lambda x: mapping.get(normalize_text(x), 0)
        )
        return result[["id_anonymat", "license_status"]]

    def classify_driving_frequency_level(self, df):
        """
        Build a driving frequency level mainly from 'frequence',
        with 'kms' used as a supporting adjustment.
        """
        results = []

        for _, row in df.iterrows():
            freq_level = parse_frequency_level(row.get("frequence", ""))
            kms_level = parse_kms_level(row.get("kms", ""))

            # Base level from reported frequency
            level = freq_level

            # Adjust with mileage information for better robustness
            if freq_level == 0 and kms_level >= 2:
                level = 1
            elif freq_level == 1 and kms_level >= 3:
                level = 2
            elif freq_level == 2 and kms_level >= 3:
                level = 3
            elif freq_level >= 3 and kms_level == 0:
                level = max(1, freq_level - 1)

            results.append({
                "id_anonymat": row["id_anonymat"],
                "driving_frequency_level": int(level)
            })

        return pd.DataFrame(results)

    def classify_driving_profile(self, df):
        """
        Build a compact driving profile feature that summarizes sparse
        driving-related background information, excluding driving frequency.
        The final profile is assigned by quantile-based binning among licensed
        participants, while non-licensed participants are fixed to 0.
        """
        results = []

        for _, row in df.iterrows():
            score = 0.0

            # License status
            license_status = 1 if normalize_text(row.get("permis", "")) == "oui" else 0
            if license_status == 1:
                score += 1

            # License categories
            license_types = normalize_text(row.get("lesquels", ""))
            license_type_score = 0.0
            if "b" in license_types:
                license_type_score += 1.0
            if "a" in license_types:
                license_type_score += 0.5
            if "am" in license_types:
                license_type_score += 0.5
            score += min(1.0, license_type_score)

            # Gearbox familiarity
            gearbox = normalize_text(row.get("boitevitesse", ""))
            if gearbox == "les deux":
                score += 2
            elif gearbox in ["manuelle", "automatique"]:
                score += 1

            # Vehicle diversity
            vehicles = normalize_text(row.get("vehicules", ""))
            if vehicles == "tous":
                score += 2
            elif vehicles not in ["", "0"]:
                vehicle_count = len([v for v in vehicles.split("-") if v.strip() != ""])
                if vehicle_count >= 2:
                    score += 2
                elif vehicle_count == 1:
                    score += 1

            # Avoidance behavior
            route_avoid = normalize_text(row.get("routesevitees", ""))
            situation_avoid = normalize_text(row.get("situationsevitees", ""))

            avoidance_count = 0
            if route_avoid not in ["", "0", "aucune"]:
                avoidance_count += len([v for v in route_avoid.split("-") if v.strip() != ""])
            if situation_avoid not in ["", "0", "aucune"]:
                avoidance_count += len([v for v in situation_avoid.split("-") if v.strip() != ""])

            if avoidance_count >= 2:
                score -= 1

            # Driving history complexity / stability
            if normalize_text(row.get("pertepoints", "")) == "oui":
                score -= 0.5
            if normalize_text(row.get("accident", "")) == "oui":
                score -= 0.5
            if normalize_text(row.get("suspension", "")) == "oui":
                score -= 1

            # Mechanical / equipment experience
            if normalize_text(row.get("CACES", "")) == "oui":
                score += 1
            if normalize_text(row.get("engins", "")) == "oui":
                score += 1

            results.append({
                "id_anonymat": row["id_anonymat"],
                "license_status": int(license_status),
                "driving_profile_score": float(score)
            })

        df_profile = pd.DataFrame(results)

        # Default profile for all participants
        df_profile["driving_profile"] = 0

        # Only licensed participants are ranked into 4 levels
        licensed_mask = df_profile["license_status"] == 1
        licensed_scores = df_profile.loc[licensed_mask, "driving_profile_score"]

        if len(licensed_scores) > 0:
            n_unique = licensed_scores.nunique()

            if n_unique >= 4:
                # Quantile-based binning into 4 ordered groups: 1, 2, 3, 4
                df_profile.loc[licensed_mask, "driving_profile"] = pd.qcut(
                    licensed_scores,
                    q=4,
                    labels=[1, 2, 3, 4],
                    duplicates="drop"
                ).astype(int)
            else:
                # Fallback when the number of unique score values is too small
                ranked = licensed_scores.rank(method="dense")
                max_rank = ranked.max()

                if max_rank > 1:
                    scaled = 1 + (ranked - 1) * 3 / (max_rank - 1)
                    df_profile.loc[licensed_mask, "driving_profile"] = scaled.round().astype(int)
                else:
                    df_profile.loc[licensed_mask, "driving_profile"] = 2

        return df_profile[["id_anonymat", "driving_profile"]]
    
    def run(self, output_name="DataPre.csv"):
        # Step 1: Load only the target sheet
        df_ia = pd.read_excel(self.file_path, sheet_name="Datas-IA-Jiajun")

        # Step 2: Clean input rows
        df_ia = self.clean_input_dataframe(df_ia)

        #print("Input columns:", df_ia.columns.tolist())
        #print(f"Valid questionnaire rows after cleaning: {len(df_ia)}")

        # Step 3: Extract selected features
        df_interface = self.classify_interface(df_ia)
        df_handness_tendency = self.classify_operational_handness_tendency(df_ia)
        df_glasses = self.classify_glasses(df_ia)
        df_license = self.classify_license_status(df_ia)
        df_frequency = self.classify_driving_frequency_level(df_ia)
        df_profile = self.classify_driving_profile(df_ia)

        # Step 4: Ensure unique ids before merge
        df_interface = self.ensure_unique_id(df_interface, "interface")
        df_handness_tendency = self.ensure_unique_id(df_handness_tendency, "handness_tendency")
        df_glasses = self.ensure_unique_id(df_glasses, "glasses")
        df_license = self.ensure_unique_id(df_license, "license")
        df_frequency = self.ensure_unique_id(df_frequency, "frequency")
        df_profile = self.ensure_unique_id(df_profile, "profile")

        # Step 5: Use cleaned ids as base table
        df_final = df_ia[["id_anonymat"]].drop_duplicates().copy()

        df_final = df_final.merge(df_interface, on="id_anonymat", how="left")
        df_final = df_final.merge(df_handness_tendency, on="id_anonymat", how="left")
        df_final = df_final.merge(df_glasses, on="id_anonymat", how="left")
        df_final = df_final.merge(df_license, on="id_anonymat", how="left")
        df_final = df_final.merge(df_frequency, on="id_anonymat", how="left")
        df_final = df_final.merge(df_profile, on="id_anonymat", how="left")

        #print("df_final columns after merge:", df_final.columns.tolist())

        # Step 6: Ensure all expected columns exist
        expected_cols = [
            "id_anonymat",
            "interface",
            "handness_tendency",
            "glasses_status",
            "license_status",
            "driving_frequency_level",
            "driving_profile"
        ]

        for col in expected_cols:
            if col not in df_final.columns and col != "id_anonymat":
                df_final[col] = 0

        df_final = df_final[expected_cols]

        # Step 7: Fill missing values with 0
        feature_cols = [col for col in df_final.columns if col != "id_anonymat"]
        df_final[feature_cols] = df_final[feature_cols].fillna(0)

        # Step 8: Rename columns with PRE_ prefix
        df_final = df_final.rename(
            columns={col: f"PRE_{col}" for col in df_final.columns if col != "id_anonymat"}
        )

        # Step 9: Save output
        self.save_path.mkdir(parents=True, exist_ok=True)
        output_file = self.save_path / output_name
        df_final.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"Questionnaire data saved to: {output_file}")
        return df_final


# ===== Example usage =====

if __name__ == "__main__":
    processor = DataPre(
        file_path=FILE_PATH,
        save_path=SAVE_PATH
    )
    processor.run()