import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils import read_data_spectrum, get_dataloader,  plot, get_input_target
from network import BiomassPredictor


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser("HyLight")
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3) 
    parser.add_argument("--type", type=str, default="custom", choices=["custom"])
    parser.add_argument("--model_path", type=str, default="model/")
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument('--test', action='store_true', default=False, help='Test the model')
    parser.add_argument("--model_seed", type=int, default=0)

    # database related arguments
    parser.add_argument("--db_server", type=str, default="x.x.x.x")
    parser.add_argument("--db", type=str, default="dbname")
    parser.add_argument("--growth_table", type=str, default="growth_data")
    parser.add_argument("--treatment_table", type=str, default="treatment")
    parser.add_argument("--raw_treatment_table", type=str, default="raw_treatment")
    return parser.parse_args()

args = parse_arguments()

def train(trainloader, model, criterion, optimizer, epoch, valloader, min_loss, cnt):
    losses = []
    for i in range(epoch):
        model.train()
        for input, target in trainloader:
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.item())
            # print(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        l = evaluate(valloader, model, criterion)
        # print("Epoch ", i, " validation loss", l)
        # save if validation loss is better than past models
        if l < min_loss:
            min_loss = l
        torch.save(model.state_dict(), args.model_path+"/nn_"+str(cnt)+".pth")
    # plt.plot(losses)
    # plt.savefig("visualization/loss"+str(i)+".png")
    # plt.close()
    return min_loss, l


def evaluate(test_loader, model, criterion):
    model.eval()
    loss = []
    for input, target in test_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss.append(criterion(output, target).cpu().detach().numpy())
    return sum(loss)

def test(test_df, model):
    model.eval()
    X_test, target = get_input_target(test_df)
   
    predicted = model(torch.from_numpy(X_test).float().cuda()).cpu().detach().numpy()
    rmse = np.sqrt(mean_squared_error(target, predicted))
    r2 = r2_score(target, predicted)
    mae = mean_absolute_error(target, predicted)
    plot(target, predicted, "BioNet", rmse, r2, "NN")
    # plot_scatter(target, predicted, "1D CNN", rmse, r2, "NN_"+str(num_band))
    return rmse, r2, mae


if __name__ == "__main__":

    results = {}
    min_loss = 100.0 
    results["rmse"] = []  
    results["r2"] = [] 
    results["valloss"] = []
    data_df = read_data_spectrum(args, 401)
    '''normalize'''
    max_val = data_df.max(axis=1).max()
    # print(max_val)
    for i in range(401):
        data_df[str(i)] /= max_val

    treatments = data_df['treatment_name'].unique().tolist()
    test_treatment = ["1", "2", "3", "4", "5", "6", "7", "8"]
    val_treatment = ["9", "10", "11", "12", "13", "14", "15", "16"]
    test_df = data_df[data_df['treatment_name'].isin(test_treatment)].copy(deep=True)
    val_df = data_df[data_df['treatment_name'].isin(val_treatment)].copy(deep=True)

    train_df = data_df[~data_df['treatment_name'].isin(test_treatment+val_treatment)].copy(deep=True)

    '''Drop treatment_name column as it is not needed for training.'''
    for df in [train_df, val_df, test_df]:
        df.drop(['treatment_name'], axis=1, inplace=True)
        df.dropna(inplace=True)

    '''For training using 100 seeds'''
    if not args.test:
        for i in range(100):
            seed_everything(i)
            train_df = shuffle(train_df)
            te_df = test_df.copy(deep=True)
            tr_df = train_df.copy(deep=True)
            va_df = val_df.copy(deep=True)
            trainloader = get_dataloader(tr_df, args.batch_size)
            # testloader = get_dataloader(test_data, args.batch_size)
            valloader = get_dataloader(va_df, args.batch_size)

            input_size = 401 + 1 # plus one for age of the plant

            model = BiomassPredictor(input_size, args.output_size).cuda()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            loss, l = train(trainloader, model, criterion, optimizer, args.epoch, valloader, min_loss, i)
            min_loss = loss if loss < min_loss else min_loss

            rmse, r2, _ = test(val_df.copy(deep=True), model)
            print(i, rmse, r2, l)
            results["rmse"].append(rmse)  
            results["r2"].append(r2)
            results["valloss"].append(l)
        result_df = pd.DataFrame.from_dict(results)
        result_df.to_csv ("results_nn.csv", index = False, header=True) 
    
    else:
        '''save best model as nn.pth manually'''
        model = BiomassPredictor(402, args.output_size).cuda()
        model.load_state_dict(torch.load(args.model_path+"/nn.pth"))
        model.eval()
        rmse, r2, mae = test(test_df.copy(deep=True), model, str(args.num_band))
        print(rmse, r2, mae)


   
