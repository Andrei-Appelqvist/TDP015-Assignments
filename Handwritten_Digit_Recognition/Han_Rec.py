import math
import mnist
import png
import random

def make_images():
    ten = range(0,10)
    values = []
    used_digit = []
    r = random.randint(1, 59000)
    training_data = mnist.read_training_data()

    for x in range(r):
        next(training_data)

    for image, dig in training_data:
        if dig in used_digit:
            continue
        else:
            for p in image:
                if p == 0:
                    values.append(255)
                else:
                    values.append(0)
            used_digit.append(dig)
            png.save_png(28, 28, values, 'digit-{}.png'.format(dig))
            values.clear()
            if len(used_digit) == 10:
                break


def train(data):
    pd = {d: 0 for d in range(10)}
    pp = {d: {p: 1 for p in range(784)} for d in range(10)}

    cnt = 0
    for deep in data:
        pd[deep[1]] += 1
        for px in deep[0]:
            if px == 1:
                pp[deep[1]][cnt] += 1
            cnt += 1
            if cnt == 784:
                cnt = 0

    a = sum(pd.values())
    for key, val in pd.items():
        pd[key] = val / a * 100

    all = sum(pp[7].values())
    cc = 0
    for k, v in pp.items():
        no_name = sum(pp[cc].values())
        #print(no_name)
        for key, value in v.items():
            v[key] = value / no_name * 100
        cc += 1
    #print(pp[7])
    return pd, pp




def predict(model, image):
    nums = {q: 0 for q in range(10)}
    pd = model[0]
    pp = model[1]
    for key, values in pd.items():
        nums[key] = math.log(pd[key])
    predict_image = image[0]
    n = image[1]

    px_cnt = 0
    nr_cnt = 0
    for nr in pp:
        for px in predict_image:
            if px == 1:
                nums[nr_cnt] += math.log(pp[nr_cnt][px_cnt])
            px_cnt += 1
            if px_cnt == 784:
                px_cnt = 0
        nr_cnt += 1
    v = max(list(nums.values()))
    for key, val in nums.items():
        if val == v:
            return [key, n]
    return 0

def evaluate(model, data):
    total = 0
    correct = 0
    for x in data:
        total += 1
        #print(x)
#        break
        t = predict(model, x)
        if t[0] == t[1]:
            correct += 1
    #print(total)
    #print(correct)
    return str(correct/total * 100) + ' %'


if __name__ == '__main__':
    print("Generating image files ...")
    make_images()
    print("Estimating the probabilities ...")
    model = train(mnist.read_training_data())
    print("Evaluating the classifier ...")
    print("Accuracy:", evaluate(model, mnist.read_test_data()))
