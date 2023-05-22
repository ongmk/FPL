import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from src.fpl.pipelines.model_pipeline.training import _encode_features

logger = logging.getLogger(__name__)
color_pal = sns.color_palette()
plt.style.use("ggplot")


def _ordered_set(input_list):
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


def evaluate_model(holdout_data, model, encoder, start_time, parameters):
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    baseline_columns = parameters["baseline_columns"]
    output_plots = {}

    # Feature Importance
    encoded_cat_cols = encoder.get_feature_names_out(
        input_features=categorical_features
    )
    features_importance = pd.DataFrame(
        data=model.feature_importances_,
        index=encoded_cat_cols.tolist() + numerical_features,
        columns=["importance"],
    )
    features_importance = features_importance.sort_values(
        by="importance", ascending=False
    ).head(10)

    ax = features_importance.sort_values("importance").plot(
        kind="barh", title="Feature Importance"
    )
    output_plots[f"{start_time}__feature_importance.png"] = ax.get_figure()

    # Holdout set evaluation
    X_holdout = holdout_data[numerical_features + categorical_features]

    X_holdout_encoded = _encode_features(
        X_holdout, categorical_features, numerical_features, encoder
    )
    holdout_predictions = model.predict(X_holdout_encoded)

    output_cols = _ordered_set(
        ["id"] + numerical_features + categorical_features + [target] + baseline_columns
    )
    output_df = holdout_data[output_cols].copy()
    eval_cols = ["prediction"] + baseline_columns
    output_df["prediction"] = holdout_predictions

    fig, axes = plt.subplots(
        nrows=1, ncols=len(eval_cols), figsize=(20, 5), sharey=True
    )

    for i, col in enumerate(eval_cols):
        output_df[f"{col}_error"] = output_df[col] - output_df[target]
        output_df[f"{col}_error"].hist(
            ax=axes[i], bins=np.arange(-3.5, 3.5, 0.1), color=color_pal[i]
        )
        mae = output_df[f"{col}_error"].abs().mean()
        axes[i].set_title(f"{col} MAE: {mae:.2f}")
        axes[i].set_xlabel(f"{col}_error")
    output_df.head()
    score = output_df["prediction_error"].abs().mean()
    logger.info(f"Model MAE: {score}")
    plt.subplots_adjust(wspace=0.1)
    output_plots[f"{start_time}__errors.png"] = fig

    output_df["start_time"] = start_time
    columns = ["start_time"] + [col for col in output_df.columns if col != "start_time"]
    output_df = output_df[columns]
    plt.close("all")

    return output_df, output_plots, score
