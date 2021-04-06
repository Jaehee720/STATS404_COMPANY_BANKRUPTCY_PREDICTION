def get_net_income_to_total_assets(input_df):
    """
    Divide net income by total assets to scale
    :param input_df: dataframe of input
    :return: net income to total assets
    Author: Jaehee Jeong
    Data: 03/18/2021
    Contact email: jjeong720@ucla.edu
    """
    a = input_df["net income"].values[0] / input_df["total assets"].values[0]
    return a


def get_cash_flow_to_total_assets(input_df):
    """
    Divide cash flow by total assets to scale
    :param input_df: dataframe of input
    :return: cash flow to total assets
    Author: Jaehee Jeong
    Data: 03/18/2021
    Contact email: jjeong720@ucla.edu
    """
    a = input_df["Cash flow"].values[0] / input_df["total assets"].values[0]
    return a


def feature_engineering(input_df):
    """
    Change columns names, drop a column and change the input value to scaled
    values
    :param input_df: dataframe of input
    :return: a clean df that can be used to run the model
    Author: Jaehee Jeong
    Data: 03/18/2021
    Contact email: jjeong720@ucla.edu
    """
    fea_eng_input_df = input_df.rename(
        columns={
            "net income": "net income to total assets",
            "Cash flow": "Cash flow to total assets",
        }
    )
    fea_eng_input_df.loc[
        0, "net income to total assets"
    ] = get_net_income_to_total_assets(input_df)
    fea_eng_input_df.loc[
        0, "Cash flow to total assets"
    ] = get_cash_flow_to_total_assets(input_df)
    fea_eng_input_df = fea_eng_input_df.drop(columns=["total assets"])
    return fea_eng_input_df
