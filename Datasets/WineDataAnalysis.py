# -*- coding: utf-8 -*-

import HeaderFile as hf

from sklearn.preprocessing import LabelEncoder

wine_data, features, target = hf.ReadData("winequality-red.csv", delimiter=";")
wine_data.info()
wine_data_dimensions = wine_data.shape
wine_data_head = wine_data.head()
wine_data_tail = wine_data.tail()
wine_data_describe = wine_data.describe()
wine_data_null_count = wine_data.isnull().sum()
wine_data_quality_count = wine_data.groupby("quality").quality.count()

hf.Plot2D("bar", wine_data_quality_count.index.values, wine_data_quality_count.values,
          title="Count of Wine Qualities",
          xlabel="Quality",
          ylabel="Count")

### Re-group qualities to reduce number of labels ###
# =============================================================================
# # Split into 2 groups - bad, good
# wine_data.loc[(wine_data["quality"] == 3) | (wine_data["quality"] == 4) | (wine_data["quality"] == 5), "quality"] = "bad"
# wine_data.loc[(wine_data["quality"] == 6) | (wine_data["quality"] == 7) | (wine_data["quality"] == 8), "quality"] = "good"
# new_wine_data_quality_count = wine_data.groupby("quality").quality.count()
# =============================================================================

# Split into 3 groups - bad, normal, good
# =============================================================================
# wine_data.loc[(wine_data["quality"] == 3) | (wine_data["quality"] == 4), "quality"] = "bad"
# wine_data.loc[(wine_data["quality"] == 5) | (wine_data["quality"] == 6), "quality"] = "normal"
# wine_data.loc[(wine_data["quality"] == 7) | (wine_data["quality"] == 8), "quality"] = "good"
# new_wine_data_quality_count = wine_data.groupby("quality").quality.count()
# =============================================================================

# =============================================================================
# # Split into 4 groups - bad, normal, good, excellent
# wine_data.loc[(wine_data["quality"] == 3) | (wine_data["quality"] == 4), "quality"] = "bad"
# wine_data.loc[(wine_data["quality"] == 5), "quality"] = "normal"
# wine_data.loc[(wine_data["quality"] == 6), "quality"] = "good"
# wine_data.loc[(wine_data["quality"] == 7) | (wine_data["quality"] == 8), "quality"] = "excellent"
# new_wine_data_quality_count = wine_data.groupby("quality").quality.count()
# =============================================================================

# =============================================================================
# quality_encoder = hf.LabelEncoder()
# wine_data["quality"] = quality_encoder.fit_transform(wine_data["quality"])
# =============================================================================

wine_data.to_csv("winequality-red-clean-v2.csv", index=False)
