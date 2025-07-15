import logging

def apply_rule_correction(row, base_tsv_col='TSV_meta'):
    """
    Apply manual rule-based adjustments to TSV prediction.
    """
    tsv = row[base_tsv_col]
    adjustment = 0.0

    fan = row.get('Fan      O/C-        3', 0)
    cooler = row.get('Evaporative cooler O/C -                4         ', 0)
    ac = row.get('Air conditioner       O/C -                5        ', 0)
    window = row.get('Window O/C -                1         ', 0)
    control_sum = fan + cooler + ac

    if fan == 1 and tsv > 1:
        adjustment -= 0.5
    if cooler == 1 and tsv > 1:
        adjustment -= 0.7
    if ac == 1 and tsv > 1:
        adjustment -= 1.0
    if str(row.get('A bit Cooler', '0')).strip() == '1' and tsv >= 0:
        adjustment -= 0.5
    if str(row.get('A bit warmer', '0')).strip() == '1' and tsv <= 0:
        adjustment += 0.5
    if row.get('HumidityPerception', 0) >= 2 and tsv > 1:
        adjustment -= 0.3
    if row.get('AirMovement', 0) <= -1 and tsv > 1:
        adjustment -= 0.3
    if control_sum == 0 and tsv < -1:
        adjustment += 0.5
    if window == 1 and tsv > 1:
        adjustment -= 0.3
    if row.get('ThermalControlIndex', 0) >= 3 and tsv > 1.5:
        adjustment -= 0.7
    if str(row.get('No Change', '0')).strip() == '1' and -0.5 <= tsv <= 0.5:
        return 0.0
    if ac == 1 and str(row.get('Much Cooler', '0')).strip() == '1':
        adjustment -= 0.3
    if row.get('Walking (Outdoor) hrs', 0) >= 2 and tsv < 0:
        adjustment += 0.5
    if row.get('CLO_score', 0) > 1.5 and tsv > 1:
        adjustment -= 0.5
    if row.get('CLO_score', 0) < 0.5 and tsv < -1:
        adjustment += 0.5

    return tsv + adjustment


def apply_rule_correction_batch(df, tsv_col='TSV_meta'):
    """
    Applies rule correction to entire DataFrame.
    Returns new column TSV_final.
    """
    logging.info('Applying rule-based correction to batch...')
    df['TSV_final'] = df.apply(lambda row: apply_rule_correction(row, base_tsv_col=tsv_col), axis=1)
    logging.info('Rule-based correction applied to batch.')
    return df
