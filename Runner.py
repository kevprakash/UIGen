import torch
from torch import optim
from torch import nn
import Utility as util
import FISSA
import UIGen
import random
import numpy as np

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def preprocessData(topNGames=500, minInteractions=2):
    print("Preprocessing Data")
    # Read, sort and remove duplicates
    c = util.readCSV()
    c = c.sort_values('Timestamp')
    c = util.removeDuplicateInteractions(c)

    # Take only top N games
    if topNGames > 0:
        c = util.getTopNofCol(c, 'Game', topNGames)

    # Iteratively remove low-interaction items
    prevLen = len(c)
    reduced = True
    while reduced:
        c = util.filterLowInteractions(c, 'Game', minInteractions)
        c = util.filterLowInteractions(c, 'ID', minInteractions)
        reduced = len(c) < prevLen
        prevLen = len(c)

    # Final Clean up
    c = util.categorize(c)
    c = c.sort_values('Timestamp')

    # Output information on new, filtered data
    print(str(len(c)) + " interactions")
    print(str(len(util.getUniqueAsList(c, 'Game'))) + " games")
    print(str(len(util.getUniqueAsList(c, 'ID'))) + " users")
    print()
    return c


def trainFISSA(data, model, epochs=5, numCandidates=2, accuracyThreshold=0.5, lr=0.0005, displayInterval=100, displayDecimalPlaces=2, genSteps=0):
    # Setting up for training
    print("\nSetting up training")
    users = util.getUniqueAsList(data, 'ID')
    print(str(len(users)) + " total users")
    numGames = len(util.getUniqueAsList(data, 'Game'))
    print(str(numGames) + " total games")
    paddedLen = util.maxDuplicate(data, 'ID') + genSteps
    model.train()

    # Loss/Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Trackers
    lossTracker = []
    accuracyTracker = []
    hitrateTracker = []

    # Training loop
    for epoch in range(epochs):
        print("\nStart of epoch " + str(epoch + 1))

        # Epoch trackers
        totalLosses = []
        totalAccuracies = []
        hits = 0.0
        intervalHits = 0.0

        # User training loop
        random.shuffle(users)
        userCount = 0
        for u in users:
            # Set up user training data
            optimizer.zero_grad()
            I = util.getUserInteractions(data, u)['Game'].tolist()
            correctIndex = I[-2]

            # Pad input vector
            unpaddedLen = len(I) - 2
            prepend = [-1 for _ in range(paddedLen - unpaddedLen)]
            prepend.extend(I[:-2])
            I = prepend
            I = torch.Tensor(util.listToOnehotMatrix(I, numGames))

            # Generate candidates to train on
            candidates = util.randomList(numCandidates-1, numGames, exlcludedValues=I)
            insertIndex = random.randint(0, len(candidates))
            candidates.insert(insertIndex, correctIndex)

            # User trackers
            index = 0
            losses = []
            accuracies = []
            hit = None
            for c in candidates:
                # Set up candidate training data
                c = util.onehot(c, numGames)
                optimizer.zero_grad()
                prediction = model(I, torch.Tensor(c).unsqueeze(0))

                # Calculate target and loss
                target = [1] if index == insertIndex else [0]
                loss = criterion(prediction[-unpaddedLen:], torch.Tensor(target).unsqueeze(0).repeat(unpaddedLen, 1))

                # Update trackers
                losses.append(loss.item())
                accuracies.append(1 if abs(prediction[-1] - target[0]) < accuracyThreshold else 0)

                # Training step
                loss.backward()
                optimizer.step()

                # Update hit tracker
                if index == insertIndex:
                    hit = abs(prediction[-1] - target[0]) < accuracyThreshold
                    hit = hit.cpu().numpy()[0]

                index += 1

            # Update hit tracker
            if hit:
                hits += 1.0
                intervalHits += 1.0

            # Calculate total losses for user
            totalLosses.append(sum(losses) / len(losses))
            totalAccuracies.append(sum(accuracies) / len(accuracies))

            # Per user display info
            if displayInterval <= 1:
                userCountString = "{:04d}".format(userCount)
                print("User " + userCountString + " -   Loss: " + "{:.{}f}".format(sum(losses) / len(losses), displayDecimalPlaces) +
                      "   |   Acc: " + "{:.{}f}".format(sum(accuracies) / len(accuracies), displayDecimalPlaces) +
                      "   |   Hit: " + str(hit))

            userCount += 1

            # Batch user display info
            if displayInterval > 1 and userCount % displayInterval == 0:
                intervalLoss = np.array(totalLosses[-displayInterval:])
                displayLoss = np.sum(intervalLoss)/intervalLoss.size
                intervalAccuracy = np.array(totalAccuracies[-displayInterval:])
                displayAccuracy = np.sum(intervalAccuracy)/intervalAccuracy.size
                userCountString = "{:04d}".format(userCount - displayInterval)
                userCountString2 = "{:04d}".format(userCount)
                print("Users " + userCountString + " to " + userCountString2 +
                      " -   Loss: " + "{:.{}f}".format(displayLoss, displayDecimalPlaces) +
                      "   |   Acc: " + "{:.{}f}".format(displayAccuracy, displayDecimalPlaces) +
                      "   |   Hit rate: " + "{:.{}f}".format(intervalHits/displayInterval, displayDecimalPlaces))
                intervalHits = 0

        # Calculate epoch final stats
        epochLosses = sum(totalLosses) / len(totalLosses)
        epochAccuracies = sum(totalAccuracies) / len(totalAccuracies)

        # Display epoch training info
        print("Epoch " + str(epoch + 1) +
              "\n     Loss: " + "{:.{}f}".format(epochLosses, displayDecimalPlaces) +
              "\n     Acc: " + "{:.{}f}".format(epochAccuracies, displayDecimalPlaces) +
              "\n     Hit rate: " + "{:.{}f}".format(hits/userCount, displayDecimalPlaces))

        # Update overall trackers
        lossTracker.append(epochLosses)
        accuracyTracker.append(epochAccuracies)
        hitrateTracker.append(hits/len(users))

    # Display final per-epoch training results
    print("Final Training stats: " +
          "\n     losses: [" + ", ".join(['%.4f' % templ for templ in lossTracker]) + "]" +
          "\n     acc: [" + ", ".join(['%.4f' % tempa for tempa in accuracyTracker]) + "]" +
          "\n     hit rate: [" + ", ".join(['%.4f' % temph for temph in hitrateTracker]) + "]")


# For testing purposes
def evaluateFISSA(data, model, numCandidates=100, topN=(5, 10, 20), displayInterval=100):
    print("Setting up for evaluation")
    users = util.getUniqueAsList(data, 'ID')
    numGames = len(util.getUniqueAsList(data, 'Game'))
    paddedLen = util.maxDuplicate(data, 'ID')
    model.eval()

    hitTracker = []

    print("Starting Evaluation")
    userIndex = -1
    for u in users:
        userIndex += 1
        predictions = []

        I = util.getUserInteractions(data, u)['Game'].tolist()
        correctIndex = I[-1]
        unpaddedLen = len(I) - 1
        prepend = [-1 for _ in range(paddedLen - unpaddedLen)]
        prepend.extend(I[:-1])
        I = prepend
        I = torch.Tensor(util.listToOnehotMatrix(I, numGames))

        candidates = util.randomList(numCandidates - 1, numGames, exlcludedValues=I)
        candidates.insert(0, correctIndex)

        for c in candidates:
            c = util.onehot(c, numGames)
            prediction = model(I, torch.Tensor(c).unsqueeze(0))
            predictions.append(prediction[0])

        ranked = np.array(predictions).argsort()
        hits = [0 in ranked[-tn:] for tn in topN]
        #p = [pred.data[0].cpu().numpy() for pred in predictions]
        #print(p)
        #print(ranked)
        hitTracker.append([1 if hit else 0 for hit in hits])

        if displayInterval <= 1:
            hitString = ["hit" if hit else "miss" for hit in hits]
            print("User " + str(userIndex) + ": " + ", ".join(hitString))
        elif (userIndex + 1) % displayInterval == 0:
            intervalHits = np.transpose(np.array(hitTracker[-displayInterval:]))
            userIndexString = "{:04d}".format(userIndex + 1 - displayInterval)
            userIndexString2 = "{:04d}".format(userIndex + 1)
            print("Users " + userIndexString + " to " + userIndexString2 + ": ")
            for i in range(len(topN)):
                intervalHitrate = sum(intervalHits[i]) / len(intervalHits[i])
                print("     Rec@" + str(topN[i]) + ": " + ("%.2f" % (intervalHitrate * 100)) + "%")


    print()
    hitTracker = np.transpose(np.array(hitTracker))
    for i in range(len(topN)):
        hitrate = sum(hitTracker[i])/len(hitTracker[i])
        print("Rec@" + str(topN[i]) + ": " + ("%.2f" % (hitrate * 100)) + "%")


def trainUIGen(data, model, epochs=5, lr=0.005, displayInterval=100, displayDecimalPlaces=2, genSteps=0):
    # Setting up training
    print("\nSetting up training")
    users = util.getUniqueAsList(data, 'ID')
    print(str(len(users)) + " total users")
    numGames = len(util.getUniqueAsList(data, 'Game'))
    print(str(numGames) + " total games")
    paddedLen = util.maxDuplicate(data, 'ID') + genSteps
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Trackers
    lossTracker = []
    accuracyTracker = []

    # Training loop
    for epoch in range(epochs):
        print("\nStart of epoch " + str(epoch + 1))

        # Epoch trackers
        totalLosses = []
        totalAccuracies = []

        # User training loop
        random.shuffle(users)
        userCount = 0
        for u in users:
            # Setting up user training data
            optimizer.zero_grad()
            I = util.getUserInteractions(data, u)['Game'].tolist()

            # User trackers
            losses = []
            accuracies = []
            # Iteratively train on subsequences, not including training/evaluation candidates
            for index in range(int(len(I)/2), len(I) - 2):
                # Subsequence data setup
                unpaddedIn = I[:index]
                prepend = [-1 for _ in range(paddedLen - len(unpaddedIn))]
                prepend.extend(unpaddedIn)
                trainIn = prepend
                trainIn = torch.Tensor(util.listToOnehotMatrix(trainIn, numGames))
                target = torch.LongTensor([I[index + 1]]).cuda()

                # Reset gradient and get prediction
                optimizer.zero_grad()
                prediction = model(trainIn)

                # Loss and accuracy calculation
                loss = criterion(prediction, target)
                losses.append(loss.item())
                accuracies.append(1 if torch.argmax(prediction) == I[index + 1] else 0)

                # Training step
                loss.backward()
                optimizer.step()

            # Update trackers
            totalLosses.append(sum(losses)/len(losses))
            totalAccuracies.append(sum(accuracies)/len(accuracies))

            # Per-user display
            if displayInterval <= 1:
                userCountString = "{:04d}".format(userCount)
                print("User " + userCountString + " -   Loss: " + "{:.{}f}".format(sum(losses) / len(losses), displayDecimalPlaces) +
                      "   |   Acc: " + "{:.{}f}".format(sum(accuracies) / len(accuracies), displayDecimalPlaces))

            userCount += 1

            # Batch user display
            if displayInterval > 1 and userCount % displayInterval == 0:
                intervalLoss = np.array(totalLosses[-displayInterval:])
                displayLoss = np.sum(intervalLoss)/intervalLoss.size
                intervalAccuracy = np.array(totalAccuracies[-displayInterval:])
                displayAccuracy = np.sum(intervalAccuracy)/intervalAccuracy.size
                userCountString = "{:04d}".format(userCount - displayInterval)
                userCountString2 = "{:04d}".format(userCount)
                print("Users " + userCountString + " to " + userCountString2 +
                      " -   Loss: " + "{:.{}f}".format(displayLoss, displayDecimalPlaces) +
                      "   |   Acc: " + "{:.{}f}".format(displayAccuracy, displayDecimalPlaces))

        # Update epoch trackers
        epochLosses = sum(totalLosses) / len(totalLosses)
        epochAccuracies = sum(totalAccuracies) / len(totalAccuracies)

        # Per-epoch display
        print("Epoch " + str(epoch + 1) +
              "\n     Loss: " + "{:.{}f}".format(epochLosses, displayDecimalPlaces) +
              "\n     Acc: " + "{:.{}f}".format(epochAccuracies, displayDecimalPlaces))

        # Update overall trackers
        lossTracker.append(epochLosses)
        accuracyTracker.append(epochAccuracies)

    # Overall per-epoch display
    print("Final Training stats: " +
          "\n     losses: [" + ", ".join(['%.4f' % templ for templ in lossTracker]) + "]" +
          "\n     acc: [" + ", ".join(['%.4f' % tempa for tempa in accuracyTracker]) + "]")


# For testing purposes
def evaluateUIGen(data, model, topN=(1, 5, 10, 20), maxGen=0, displayInterval=100):
    print("Setting up for evaluation")
    users = util.getUniqueAsList(data, 'ID')
    numGames = len(util.getUniqueAsList(data, 'Game'))
    paddedLen = util.maxDuplicate(data, 'ID') + maxGen
    model.eval()

    print("Starting Evaluation")
    userIndex = -1
    accTracker = []
    for u in users:
        userIndex += 1
        I = util.getUserInteractions(data, u)['Game'].tolist()

        unpaddedIn = I[:-1]
        prepend = [-1 for _ in range(paddedLen - len(unpaddedIn))]
        prepend.extend(unpaddedIn)
        evalIn = prepend
        evalIn = torch.Tensor(util.listToOnehotMatrix(evalIn, numGames))
        target = torch.LongTensor([I[-1]]).cuda()

        prediction = model(evalIn)
        accuracies = [1 if target in torch.topk(prediction, n)[1] else 0 for n in topN]
        accTracker.append(accuracies)

        if displayInterval <= 1:
            hitString = ["hit" if a == 1 else "miss" for a in accuracies]
            print("User " + str(userIndex) + ": " + ", ".join(hitString))
        elif (userIndex + 1) % displayInterval == 0:
            intervalAccs = np.transpose(np.array(accTracker[-displayInterval:]))
            userIndexString = "{:04d}".format(userIndex + 1 - displayInterval)
            userIndexString2 = "{:04d}".format(userIndex + 1)
            print("Users " + userIndexString + " to " + userIndexString2 + ": ")
            for i in range(len(topN)):
                intervalHitrate = sum(intervalAccs[i]) / len(intervalAccs[i])
                print("     Rec@" + str(topN[i]) + ": " + ("%.2f" % (intervalHitrate * 100)) + "%")

    print()
    hitTracker = np.transpose(np.array(accTracker))
    print("Final Evaluation Scores:")
    for i in range(len(topN)):
        hitrate = sum(hitTracker[i]) / len(hitTracker[i])
        print("     Rec@" + str(topN[i]) + ": " + ("%.2f" % (hitrate * 100)) + "%")


def evaluateCombined(data, model, genModel, genSteps, maxGenSteps, numCandidates=100, topN=(1, 5, 10, 20), displayInterval=100):
    # Setting up evaluation data
    print("Setting up for evaluation")
    users = util.getUniqueAsList(data, 'ID')
    numGames = len(util.getUniqueAsList(data, 'Game'))
    paddedLen = util.maxDuplicate(data, 'ID') + maxGenSteps
    # Set models to eval mode
    model.eval()
    genModel.eval()

    # Tracker
    hitTracker = []
    finalHitrates = []

    try:
        # Evaluation loop
        print("Starting Evaluation")
        userIndex = -1
        for u in users:
            # Setting up user trackers
            userIndex += 1
            predictions = []

            # Setting up user eval data
            I = util.getUserInteractions(data, u)['Game'].tolist()
            correctIndex = I[-1]

            # Padding input
            unpaddedLen = len(I) - 1
            prepend = [-1 for _ in range(paddedLen - unpaddedLen)]
            prepend.extend(I[:-1])
            I = prepend
            I = torch.Tensor(util.listToOnehotMatrix(I, numGames))

            # Generative input steps
            for g in range(genSteps):
                gen = genModel(I)
                I = torch.cat([I[1:], gen], dim=0)

            # Generating candidates
            candidates = util.randomList(numCandidates - 1, numGames, exlcludedValues=I)
            candidates.insert(0, correctIndex)

            # Per-candidate prediction
            for c in candidates:
                c = util.onehot(c, numGames)
                prediction = model(I, torch.Tensor(c).unsqueeze(0))
                predictions.append(prediction[0])

            # Calculating hits
            ranked = np.array(predictions).argsort()
            hits = [0 in ranked[-tn:] for tn in topN]
            hitTracker.append([1 if hit else 0 for hit in hits])

            # Displaying user eval data
            if displayInterval <= 1:
                hitString = ["hit" if hit else "miss" for hit in hits]
                print("User " + str(userIndex) + ": " + ", ".join(hitString))
            elif (userIndex + 1) % displayInterval == 0:
                intervalHits = np.transpose(np.array(hitTracker[-displayInterval:]))
                userIndexString = "{:04d}".format(userIndex + 1 - displayInterval)
                userIndexString2 = "{:04d}".format(userIndex + 1)
                print("Gen " + str(genSteps) + " Users " + userIndexString + " to " + userIndexString2 + ": ")
                for i in range(len(topN)):
                    intervalHitrate = sum(intervalHits[i]) / len(intervalHits[i])
                    print("     Rec@" + str(topN[i]) + ": " + ("%.2f" % (intervalHitrate * 100)) + "%")

    finally:
        # Displaying final evaluation data
        print()
        hitTracker = np.transpose(np.array(hitTracker))
        print("Final Evaluation Scores for " + str(genSteps) + " generation steps: ")
        # Calculating rec@N values
        for i in range(len(topN)):
            hitrate = sum(hitTracker[i]) / len(hitTracker[i])
            print("Rec@" + str(topN[i]) + ": " + ("%.2f" % (hitrate * 100)) + "%")
            finalHitrates.append(hitrate)

        return finalHitrates


# For testing purposes
def runFISSA(train=True, evaluate=False):
    data = preprocessData(minInteractions=5, topNGames=500)
    model = FISSA.FISSA(
        I=len(util.getUniqueAsList(data, 'Game')),
        L=util.maxDuplicate(data, 'ID'),
        d=2**6,
        B=5,
        activation=nn.Sigmoid()
    )
    if train:
        trainFISSA(data, model, epochs=5, displayInterval=100)
    if evaluate:
        evaluateFISSA(data, model, numCandidates=100, topN=(1, 5, 10, 20), displayInterval=100)


# For testing purposes
def runUIGen(train=True, evaluate=False):
    data = preprocessData(minInteractions=5, topNGames=500)

    model = UIGen.UIGen(
        I=len(util.getUniqueAsList(data, 'Game')),
        L=util.maxDuplicate(data, 'ID'),
        d=2 ** 6,
    )
    if train:
        trainUIGen(data, model, epochs=5, displayInterval=100, lr=0.0005)
    if evaluate:
        evaluateUIGen(data, model, topN=(1, 5, 10, 20), displayInterval=100)


def runCombined(train=True, evaluate=False, genSteps=(0, 1, 2, 3), topN=(1, 5, 10, 25, 50, 100), sizeLimited=True, topGames=1500):
    topGamestoLR = {
        500: 0.0005,
        1000: 0.00005,
        1500: 0.00005,
    }
    topGamestoDisplayInterval = {
        500: 100,
        1000: 100,
        1500: 250
    }
    topGamestoD={
        500: 2**6,
        1000: 2**6,
        1500: 2**7
    }
    data = preprocessData(minInteractions=5, topNGames=topGames if sizeLimited else -1)
    I = len(util.getUniqueAsList(data, 'Game'))
    L = util.maxDuplicate(data, 'ID') + max(genSteps)
    d = topGamestoD[topGames] if sizeLimited else 2 ** 8

    model = FISSA.FISSA(I=I, L=L, d=d, B=5, activation=nn.Sigmoid())
    genModel = UIGen.UIGen(I=I, L=L, d=d)

    dI = topGamestoDisplayInterval[topGames] if sizeLimited else 1000

    if train:
        lr = topGamestoLR[topGames] if sizeLimited else 0.00001
        trainFISSA(data, model, epochs=5, displayInterval=dI, lr=lr, genSteps=max(genSteps))
        trainUIGen(data, genModel, epochs=5, displayInterval=dI, lr=lr, genSteps=max(genSteps))

    if evaluate:
        evaluateUIGen(data, genModel, topN=topN, displayInterval=dI, maxGen=max(genSteps))
        accuracies = []
        try:
            for gs in genSteps:
                acc = evaluateCombined(data, model, genModel, gs, max(genSteps), numCandidates=100, topN=topN, displayInterval=dI)
                accuracies.append(acc)

        finally:
            print("\nFinal overall evaluation:")
            for a in range(len(accuracies)):
                print("     " + str(genSteps[a]) + " generation steps: ")
                for i in range(len(topN)):
                    print("          Rec@" + str(topN[i]) + ": " + ("%.2f" % (accuracies[a][i] * 100)) + "%")


runCombined(train=True, evaluate=True, sizeLimited=True)