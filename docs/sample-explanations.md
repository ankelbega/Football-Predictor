# Example LightGBM explanations

Generated from the validation season with a model trained on earlier seasons.
SHAP values are contributions to the selected class raw score, not probability points.

## Example 1: Burnley vs Man City

Most likely outcome: **away win**.

Probabilities: home 12.3%, draw 15.9%, away 71.8%.

Top contributing features:
- `Elo home advantage over away`: raised selected-class raw score (+0.7870); Elo home advantage over away (is -249) increased the model score for the elo-class prediction.
- `away team's rest`: raised selected-class raw score (+0.1815); away team's rest (is 75 days) increased the model score for the away-class prediction.
- `away overall last10 goals against per match`: raised selected-class raw score (+0.0655); away overall last10 goals against per match (is 0.70 per match) increased the model score for the away-class prediction.
- `away overall last10 goals for per match`: raised selected-class raw score (+0.0289); away overall last10 goals for per match (is 2.30 per match) increased the model score for the away-class prediction.
- `away away last10 goals for per match`: raised selected-class raw score (+0.0276); away away last10 goals for per match (is 1.90 per match) increased the model score for the away-class prediction.

## Example 2: Arsenal vs Nott'm Forest

Most likely outcome: **home win**.

Probabilities: home 69.8%, draw 16.8%, away 13.4%.

Top contributing features:
- `Elo home advantage over away`: raised selected-class raw score (+0.6070); Elo home advantage over away (is 252) increased the model score for the elo-class prediction.
- `home home last10 goals for per match`: raised selected-class raw score (+0.1445); home home last10 goals for per match (is 2.80 per match) increased the model score for the home-class prediction.
- `home overall last10 goals for per match`: raised selected-class raw score (+0.0746); home overall last10 goals for per match (is 2.20 per match) increased the model score for the home-class prediction.
- `home team's home points form`: lowered selected-class raw score (-0.0350); home team's home points form (is 2.00 points per match) decreased the model score for the home-class prediction.
- `away overall last10 goals for per match`: raised selected-class raw score (+0.0225); away overall last10 goals for per match (is 1.50 per match) increased the model score for the away-class prediction.

## Example 3: Bournemouth vs West Ham

Most likely outcome: **away win**.

Probabilities: home 36.9%, draw 20.8%, away 42.3%.

Top contributing features:
- `Elo home advantage over away`: raised selected-class raw score (+0.2357); Elo home advantage over away (is 8) increased the model score for the elo-class prediction.
- `away team's rest`: raised selected-class raw score (+0.1546); away team's rest (is 76 days) increased the model score for the away-class prediction.
- `season progress`: raised selected-class raw score (+0.0747); season progress (is 0.0263) increased the model score for the season-class prediction.
- `home team's recent points form`: lowered selected-class raw score (-0.0332); home team's recent points form (is 0.60 points per match) decreased the model score for the home-class prediction.
- `home home last5 goals against per match`: lowered selected-class raw score (-0.0310); home home last5 goals against per match (is 2.20 per match) decreased the model score for the home-class prediction.
