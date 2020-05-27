import linRegpipe
from tabulate import tabulate

targets = ['TT','DT','TD','PERF','CPU','MEM']
techniques = ['PRE','THR','DLTC','DTC','POST']
p = linRegpipe.Pipeline()

for tar in targets:
    for tech in techniques:
        print(tech, tar)
        result = []
        ex = 10
        for i in range(ex):
            ans = p.runpipeline(target=tar, technique=tech)
            # y_test.mean()----y_pred.mean()----abs(y_pred.mean()-y_test.mean())----MAE----MAPE
            result.append(ans)
            #print(ans)

        #print(result)
        print(tabulate(result,
                       headers=['y_test.mean()', 'y_pred.mean()', 'abs(y_pred.mean()-y_test.mean())', 'MAE', 'MRE']))

        mre = 0
        mae = 0
        for r in result:
            mre += r[-1]
            mae += r[3]
        mre /= ex
        mae /= ex
        print("mean of MRE", mre)
        print("mean of MAE", mae)
