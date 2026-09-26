# eda_database.py

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- SOTA Dependencies ---
try:
    from database import db_manager

    DB_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import database or config module: {e}")
    DB_AVAILABLE = False

try:
    from unified_ingredient_parser import parse_ingredient_simple

    PARSER_AVAILABLE = True
except ImportError:
    print(
        (
            "Warning: 'ingredient-parser' or 'ingredient_normalizer' not found. Analysis will be "
            "less "
            "accurate."
        )
    )
    print(
        (
            "         Ensure 'ingredient-parser-nlp' is installed with: pip install "
            "ingredient-parser-nlp"
        )
    )
    PARSER_AVAILABLE = False

# --- Configuration & Constants ---
TOP_N_ITEMS = 15
KETO_CARB_THRESHOLD_G = 25

# --- Schema Alignment ---
COL_CALORIES = "calories"
COL_PROTEIN = "protein_g"
COL_FAT = "total_fat_g"
COL_CARB = "carbohydrate_g"

# --- Plotting Style ---
sns.set_theme(style="whitegrid", palette="viridis")
FIG_SIZE_WIDE = (18, 8)
FIG_SIZE_SQUARE = (14, 12)


# --- Data Loading Functions (Unchanged) ---
def load_nutrition_data_from_db() -> pd.DataFrame | None:
    print("--- Loading Nutrition Data from SQLite database ---")
    try:
        with db_manager.get_sqlite_connection() as conn:
            df = pd.read_sql("SELECT * FROM nutrition_facts", conn)

        required_macros = [COL_PROTEIN, COL_FAT, COL_CARB, COL_CALORIES]
        missing_cols = [col for col in required_macros if col not in df.columns]
        if missing_cols:
            print(
                f"Error: The 'nutrition_facts' table is missing required columns: {missing_cols}"
            )
            return df

        df["calculated_calories"] = (
            (df[COL_PROTEIN] * 4) + (df[COL_CARB] * 4) + (df[COL_FAT] * 9)
        )
        correlation = df[COL_CALORIES].corr(df["calculated_calories"])
        print(
            (
                f"Calorie Sanity Check: Correlation between reported and calculated calories: "
                f"{correlation:.4f}"
            )
        )

        print(f"Successfully loaded {len(df)} records from 'nutrition_facts'.\n")
        return df
    except Exception as e:
        print(f"Error: Failed to load data from 'nutrition_facts' table: {e}")
        return None


def load_recipes_from_opensearch(limit: int = 1000) -> pd.DataFrame | None:
    print("--- Loading Recipe Data from OpenSearch ---")
    client = db_manager.get_opensearch_client()
    if not client:
        print("Error: OpenSearch client is not available. Cannot load recipes.\n")
        return None

    try:
        query = {"query": {"match_all": {}}}
        response = client.search(index="recipes", body=query, size=limit)

        hits = response["hits"]["hits"]
        if not hits:
            return pd.DataFrame()

        recipes_data = [hit["_source"] for hit in hits]
        df = pd.DataFrame.from_records(recipes_data)

        print(f"Successfully loaded {len(df)} recipes from OpenSearch.\n")
        return df
    except Exception as e:
        print(f"Error: Failed to load data from OpenSearch 'recipes' index: {e}")
        return None


# --- Analysis & Plotting Functions (Unchanged) ---
def plot_nutrition_db_analysis(df: pd.DataFrame):
    print("--- Part 1: Analysis of the Nutrition Database ---")
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_SQUARE)
    fig.suptitle("Analysis of the Nutrition Database", fontsize=20, weight="bold")
    if COL_CALORIES in df.columns and "calculated_calories" in df.columns:
        sns.scatterplot(
            data=df, x=COL_CALORIES, y="calculated_calories", alpha=0.3, ax=axes[0, 0]
        )
        axes[0, 0].plot(
            [0, df[COL_CALORIES].max()],
            [0, df[COL_CALORIES].max()],
            "r--",
            label="Ideal (y=x)",
        )
        axes[0, 0].legend()
    axes[0, 0].set_title("Data Quality: Reported vs. Calculated Calories")
    macros = [COL_PROTEIN, COL_FAT, COL_CARB]
    available_macros = [m for m in macros if m in df.columns]
    if available_macros:
        sns.boxplot(data=df[available_macros], ax=axes[0, 1], palette="mako")
    axes[0, 1].set_title("Distribution of Macronutrients (per 100g)")
    if all(col in df.columns for col in [COL_CARB, COL_FAT, COL_PROTEIN]):
        low_carb_foods = df[df[COL_CARB] <= 5]
        sns.scatterplot(
            data=low_carb_foods, x=COL_FAT, y=COL_PROTEIN, alpha=0.5, ax=axes[1, 0]
        )
    axes[1, 0].set_title("Protein vs. Fat Profile of Low-Carb Foods (<5g)")
    numeric_cols_for_pca = df.select_dtypes(include=np.number).columns.drop(
        [COL_CALORIES, "calculated_calories"], errors="ignore"
    )
    if not numeric_cols_for_pca.empty:
        X_scaled = StandardScaler().fit_transform(df[numeric_cols_for_pca].values)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(data=components, columns=["PC1", "PC2"])
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", alpha=0.3, ax=axes[1, 1])
        axes[1, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        axes[1, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    axes[1, 1].set_title('PCA of Nutrients: A "Food Map"')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- NLP Analysis and Plotting Function ---


def infer_and_analyze_diets(df_recipes: pd.DataFrame):
    """
    Performs NLP analysis on recipe ingredients to find common terms and visualizes the results.
    """
    if not PARSER_AVAILABLE:
        print(
            (
                "Cannot perform NLP analysis because 'ingredient-parser' is not installed or "
                "normalizer is "
                "unavailable."
            )
        )
        return

    print("\n--- Part 2 & 3: NLP Analysis of Recipe Ingredients ---")

    all_ingredients_flat = []
    for _, recipe in df_recipes.iterrows():
        for ing_string in recipe.get("ingredients", []):
            # Use the normalizer to get a clean, basic ingredient name
            cleaned_ing_name = parse_ingredient_simple(ing_string)
            if cleaned_ing_name:
                all_ingredients_flat.append(cleaned_ing_name)

    # Calculate top N most common ingredients
    top_common_ingredients = Counter(all_ingredients_flat).most_common(TOP_N_ITEMS)

    print(f"Total unique ingredients processed: {len(set(all_ingredients_flat))}")
    print(f"Top {TOP_N_ITEMS} most common ingredients: {top_common_ingredients}")

    # --- New Visualizations ---
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)  # Adjust subplot layout

    # Plot 1: Top N Most Common Ingredients
    if top_common_ingredients:
        df_top_ingredients = pd.DataFrame(
            top_common_ingredients, columns=["ingredient", "count"]
        )
        sns.barplot(
            data=df_top_ingredients,
            x="count",
            y="ingredient",
            hue="ingredient",
            ax=axes[0],
            palette="viridis",
            legend=False,
        )
        axes[0].set_title(f"Top {TOP_N_ITEMS} Most Common Ingredients")
        axes[0].set_xlabel("Count in Recipes")
        axes[0].set_ylabel("Ingredient")
    else:
        axes[0].text(
            0.5,
            0.5,
            "No ingredients found for analysis.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0].set_title(f"Top {TOP_N_ITEMS} Most Common Ingredients (No Data)")

    # Plot 2: Distribution of Unique Ingredient Count Per Recipe
    unique_ingredient_counts = []
    for _, recipe in df_recipes.iterrows():
        ingredients_in_recipe = [
            parse_ingredient_simple(ing_string)
            for ing_string in recipe.get("ingredients", [])
            if parse_ingredient_simple(ing_string)
        ]
        unique_ingredient_counts.append(len(set(ingredients_in_recipe)))

    if unique_ingredient_counts:
        sns.histplot(
            unique_ingredient_counts,
            bins=range(
                min(unique_ingredient_counts), max(unique_ingredient_counts) + 2
            ),
            kde=True,
            ax=axes[1],
        )
        axes[1].set_title("Distribution of Unique Ingredient Count Per Recipe")
        axes[1].set_xlabel("Number of Unique Ingredients")
        axes[1].set_ylabel("Number of Recipes")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No unique ingredient count data for analysis.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1].set_title(
            f"Top {TOP_N_ITEMS} Most Common Ingredient Bigrams (No Data)"
        )  # Note: Title remains for bigrams if no unique ingredient counts

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Print key results for the markdown report ---
    print("\n--- Key Findings for Report (NLP Analysis) ---")
    print(f"Total recipes analyzed: {len(df_recipes)}")
    if top_common_ingredients:
        print(
            (
                f"Most common ingredient: '{top_common_ingredients[0][0]}' (appears "
                f"{top_common_ingredients[0][1]} "
                f"times)"
            )
        )
    else:
        print("No common ingredients found.")
    if unique_ingredient_counts:
        print(
            f"Average unique ingredients per recipe: {np.mean(unique_ingredient_counts):.2f}"
        )
        print(
            f"Median unique ingredients per recipe: {np.median(unique_ingredient_counts)}"
        )
    else:
        print("No unique ingredient count statistics.")


def main():
    """Main function to run the full EDA pipeline using the database layer."""

    if not DB_AVAILABLE:
        print("\nDatabase modules not available. Aborting EDA.")
        return

    print("=" * 80)
    print("Starting State-of-the-Art Exploratory Data Analysis (Database Mode)")
    print("=" * 80)

    df_nutrition = load_nutrition_data_from_db()
    if df_nutrition is not None:
        plot_nutrition_db_analysis(df_nutrition)

    df_recipes = load_recipes_from_opensearch()
    if df_recipes is not None and not df_recipes.empty:
        infer_and_analyze_diets(df_recipes)
    else:
        print("\nSkipping recipe analysis because no recipe data was loaded.")


if __name__ == "__main__":
    main()
