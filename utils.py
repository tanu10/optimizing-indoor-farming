from scipy.sparse import data
from sqlalchemy import create_engine
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from network import BiomassPredictor

def create_db_engine(ip, db):
    return create_engine("mysql+pymysql://user:password@"+ip+":3306/"+db)


'''read growth data and add customized spectrum, raw_treatment table contains the full spectrum info.
    num: number of equidistant bands. '''
def read_data_spectrum(args, num):
    engine = create_db_engine(args.db_server, args.db)
    # change batch_id if needed
    growth_df = pd.read_sql("SELECT batch_id, treatment_name, date, digital_biomass FROM " + args.growth_table , con=engine)
    treatment_df = pd.read_sql("SELECT * FROM " + args.raw_treatment_table, con=engine)

    col_names = ['treatment_name']
    for i in range(num):
        col_names.append(str(i))
    spec_df = pd.DataFrame(columns = col_names)
    spec_df['treatment_name'] = treatment_df['treatment_name']
    num_wavelength_per_band = 401//num
    for i in range(num):
        list_name = [str(j) for j in range(i*num_wavelength_per_band,(i+1)*num_wavelength_per_band)]
        spec_df[str(i)] = treatment_df[list_name].sum(axis=1)

    compute_age = lambda x: ((x - min(x)).dt.days + 1)
    growth_df['age'] = growth_df.groupby('batch_id')['date'].transform(compute_age)
    print(growth_df['age'].max())
    growth_df['age'] /= growth_df['age'].max()
    growth_df['digital_biomass'] /= growth_df['digital_biomass'].max()

    growth_df.drop(['batch_id', 'date'], axis=1, inplace=True)
    growth_df = pd.merge(growth_df, spec_df, how='left', on='treatment_name')

    return growth_df

def get_dataloader(data_df, batch):    
    target = data_df.pop('digital_biomass').to_frame()
    target = torch.from_numpy(target.values.astype(np.float32))
    input = torch.from_numpy(data_df.values.astype(np.float32))
    train = TensorDataset(input, target)
    train_loader = DataLoader(train, batch_size=batch, shuffle=True)
    return train_loader

def get_input_target(data_df):
    target = data_df.pop('digital_biomass').to_frame()
    # print(data_df.columns, target.columns)
    return data_df.values.astype(np.float32), target.values.astype(np.float32)


def test_spectrum():
    '''401=age+401 values for wavelengths'''
    model = BiomassPredictor(402, 1)
    model.load_state_dict(torch.load("model/nn.pth"))
    model.eval()
    
    df = pd.read_csv("spectrum.csv", header=None)
    spec_list = df.T.values.tolist()
    for spec in spec_list:
        '''Insert age value (1.0) at the start of the input vector'''
        spec.insert(0, 1.0)
    print(model(torch.tensor([spec])).detach())

def plot(y_test, preds, rtype, mse, r2, fn):
    x = [i for i in range(len(preds))]
    plt.plot(y_test, alpha=0.6, color='b', label="Ground Truth")
    plt.plot(x, preds, alpha=0.6, color='r',label="Prediction")
    plt.title(rtype, weight='bold', fontsize=16)#+'(MSE=%1.3f' %mse +', R$^2$=%1.3f)' %r2)
    # plt.title(rtype+'(Band, R$^2$=%1.3f)' %r2)
    plt.xlabel("Plant Samples", weight='bold', fontsize=15)
    plt.ylabel("Normalized Biomass", weight='bold', fontsize=15)
    plt.legend()
    ax = plt.gca()
    ax.set_axisbelow(True)

    ax.grid(color="lightgrey")
    ax.set_facecolor("snow")
    yloc = plticker.MultipleLocator(base=0.1)
    xloc = plticker.MultipleLocator(base=500)
    ax.yaxis.set_major_locator(yloc)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_major_locator(xloc)
    #ax.set_xticks([0, 1, 2])

    plt.savefig("visualization/"+fn+".png")
    plt.show()
    plt.close()


def plot_scatter(y_test, preds, rtype, mse, r2, fn):
    x = [i for i in range(len(preds))]
    plt.scatter(y_test, preds, color='k', s=5)
    plt.title(rtype, weight='bold', fontsize=16)
    # plt.title(rtype+'(Band, R$^2$=%1.3f)' %r2)
    plt.xlabel("Ground Truth", weight='bold', fontsize=15)
    plt.ylabel("Prediction", weight='bold', fontsize=15)
    plt.legend()
    ax = plt.gca()
    ax.set_axisbelow(True)

    ax.grid(color="lightgrey")
    ax.set_facecolor("snow")
    yloc = plticker.MultipleLocator(base=0.1)
    xloc = plticker.MultipleLocator(base=0.1)
    ax.yaxis.set_major_locator(yloc)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_major_locator(xloc)
    #ax.set_xticks([0, 1, 2])

    plt.savefig("visualization/"+fn+"_scatter.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    test_spectrum()


