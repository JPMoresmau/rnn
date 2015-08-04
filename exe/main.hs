{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
-- | Main entry point for executable, mainly for tests
module Main (main) where

import AI.Network.RNN
import AI.GeneticAlgorithm.Simple
import Control.Monad
import Control.Monad.Random hiding (fromList)
import qualified Data.Text as T
import Data.IORef
import System.Directory
--import Debug.Trace
import System.Environment

import Numeric.LinearAlgebra.HMatrix

main :: IO()
main = do
    -- txt <- liftM (T.take 100) $ T.readFile "data/tinyshakespeare.txt"
    let txt= "hello world!"
        trainData = train txt
        fn = "lstm.learn1.helloworld"
        generateLength = min 50 (T.length txt)
    ex <- doesFileExist fn
    args <- getArgs
    rnn<-case args of
        ("gradient":mg:_) -> do
            let maxGen = read mg
            rd <- readEx fn ex trainData
            gradient rd generateLength maxGen
        ("rmsprop":mg:_) -> do
            let maxGen = read mg
            rd <- readEx fn ex trainData
            rmsprop rd generateLength maxGen
        ("genetic":mg:_) -> do
            let maxGen = read mg
            b <- if ex
               then do
                 rnn1::LSTMNetwork <- read <$> readFile fn
                 return $ buildExisting rnn1 trainData
              else return $ buildTD trainData
            genetic b generateLength maxGen
        _ -> error "rnn gradient <maxgen> | rmsprop <maxgen> | generic <maxgen>"
    writeFile fn $ show rnn
    where
      readEx fn ex trainData =
          if ex
                then do
                  r <- read <$> readFile fn
                  return $ RNNData r trainData (fromIntegral $ length $ tdInputs trainData)
                else
                  evalRandIO $ buildTD trainData
      (train, gener) = (textToTrainData,generate)
      genStep maxGen = maxGen `div` 10
      gradient (RNNData r td _) generateLength maxGen = learnGradientDescent r td $ test generateLength maxGen
      rmsprop (RNNData r td _) generateLength maxGen = learnRMSProp r td $ test generateLength maxGen
      genetic r generateLength maxGen = do
        fitnessList <- newIORef []
        (RNNData rnn _ _) <- runGAIO 64 0.1 r $ stopf2 generateLength maxGen fitnessList
        return rnn
      test generateLength maxGen rnn td gen= do
        when (mod gen (genStep maxGen) == 0) $ do
            let c= cost rnn td
            print (show gen ++ ":" ++ show c)
            t <- evalRandIO $ gener (tdData td) generateLength (size $ head $ tdInputs td) rnn
            print t
        return $ gen < maxGen
      stopf2 generateLength maxGen fs rd@(RNNData rnn td _) gen = do
        _ <- test generateLength maxGen rnn td gen
        stopf fs maxGen rd gen

-- | Build LSTM network
buildTD ::(Monad m,RandomGen g) =>  TrainData a Int -> RandT g m (RNNData LSTMNetwork Int a)
buildTD td = buildNetworkData td lstmFullSize

-- | Build from existing data for genetic algorithm, to restart from where we were
buildExisting :: (Monad m,RandomGen g) =>  LSTMNetwork -> TrainData a Int -> RandT g m (RNNData LSTMNetwork Int a)
buildExisting rnn1 td@(TrainData is _ _ _) = do
  g <- getSplit
  let rnn2 = evalRand (mutateNetwork rnn1) g
  return $ RNNData rnn2 td (fromIntegral $ length is)

-- DRAGONS
--
--main :: IO()
--main = do
--   -- txt <- T.readFile "data/tinyshakespeare.txt"
--    let txt= "hello world!"
--        (is,os,m) = textToTrainData txt
--        sz = DM.size m
--        fn = "lstm.learn1.helloworld"
--    print sz
--    print $ length is
--    print $ length os
--    ex <- doesFileExist fn
--    (RNNData r _ _ _) <- if ex
--        then do
--          r <- read <$> readFile fn
--          return $ RNNData r [] [] 0
--        else
--          evalRandIO $ build sz is os
--    let ls = toList $ toVector r
--    --print ls
--    --print $ toList $ chead ocs
--    --print ls
--    --print is
--    --print $ lstmList sz ls (toList $ head is)
--    --print $ mapAccumL (lstmList sz) ls (map toList is)
--    --headJet $ tailJet $ jet $ grads
----    let c= cost sz (fromIntegral $ length is) (map (toList) is) (map (toList) os) ls
----    print c
----    let gs= grad (cost sz (auto $ fromIntegral $ length is) (map (map auto . toList) is) (map (map auto . toList) os)) ls
----    --let gs=jacobian (lstmList (lstmFullSize sz) (map auto ls)) (toList $ head os)
----    -- print gs
----    let ls2 = zipWith (\l g->l-g*0.1) ls gs
----    let c2= cost sz (fromIntegral $ length is) (map (toList) is) (map (toList) os) ls2
----    print c2
----    where
----        gFun :: [AD s (Sparse Double)]
----                   -> AD s (Sparse Double)
----        gFun =  map sigmoid
--    let (l,lis,los)=((fromIntegral $ length is),(map toList is),(map toList os))
--    -- lsf <- learn sz ls l lis los 0
--    let myl= length ls
--    print myl
--    -- let adCost = cost sz (auto l) (map (map auto) lis) (map (map auto) los)
--    -- lsf <- learn2 sz ls (replicate myl 0) (replicate myl 0) (replicate myl 0) l lis los 0 (m,(min 50 (T.length txt)))
--    lsf <- learn sz ls l lis los 0 (m,(min 50 (T.length txt)))
--    --print lsf
--    let rnn::LSTMNetwork = fromVector sz (fromList lsf)
--    writeFile fn $ show rnn
--    --print rnn
--    -- print $ length is
--    let s = size $ head is
----    -- print s
----    let tl= T.length txt
----    -- print tl
----    -- print m
----    let c= cost sz l lis los ls
----    print c
----    let c2= cost sz l lis los lsf
----    print c2
--    t <- evalRandIO $ generate m (min 50 (T.length txt)) s rnn
--    print t
--    return ()
--    where
--          test sz ls l is os gen (m,gl) = when ((mod gen 500) == 0) $ do
--                let c= cost sz l is os ls
--                print (show gen ++ ":" ++ show c)
--                let rnn::LSTMNetwork = fromVector sz (fromList ls)
--                t <- evalRandIO $ generate m gl (length $ head is) rnn
--                print t
--          learn sz ls l is os gen testInfo = do
--            test sz ls l is os gen testInfo
--            if gen < 10000
--                then do
--                    let ls2 = learnGradientDescent (cost sz (auto l) (map (map auto) is) (map (map auto) os)) ls
--                    --let gs= grad (cost sz (auto l) (map (map auto) is) (map (map auto) os)) ls
--                    --    ls2 = zipWith (\o g->o-g*0.1) ls gs
--                    learn sz ls2 l is os (gen + 1) testInfo
--                else return ls
--          learn2 sz ls rgs rgs2 ugs l is os gen testInfo= do
--            test sz ls l is os gen testInfo
--            if gen < 10000
--                then do
--                    let gs= force $ grad (cost sz (auto l) (map (map auto) is) (map (map auto) os)) ls
--                        rgup = force $ zipWith (\rg g-> 0.95 * rg + 0.05 * g) rgs ls
--                        rg2up = force $ zipWith (\rg2 g-> 0.95 * rg2 + 0.05 * (g ** 2)) rgs2 ls
--                        ugup = force $ zipWith4 (\ud zg rg rg2 -> 0.9 * ud - 1e-4 * zg / sqrt(rg2 - rg ** 2 + 1e-4)) ugs gs rgup rg2up
--                        -- ls2 = zipWith (\o g->o-g*0.1) ls gs
--                        ls2 = force $ zipWith (+) ls ugup
--                    learn2 sz ls2 rgup rg2up ugup l is os (gen + 1) testInfo
--                else return ls
--          learn3 _ ls _ _ _ _ [] [] _ = return ls
--          learn3 sz ls rgs rgs2 ugs l is os gen = do
--            let batchSize = 100
--                (is1,is2) = splitAt batchSize is
--                (os1,os2) = splitAt batchSize os
--            when ((mod gen 100) == 0) $ do
--                let c= cost sz l is1 os1 ls
--                print (show gen ++ ":" ++ show c)
--            --if gen < 1
--            --    then do
--            let gs= force $ grad (cost sz (auto l) (map (map auto) is1) (map (map auto) os1)) ls
--                rgup = force $ zipWith (\rg g-> 0.95 * rg + 0.05 * g) rgs ls
--                rg2up = force $ zipWith (\rg2 g-> 0.95 * rg2 + 0.05 * (g ** 2)) rgs2 ls
--                ugup = force $ zipWith4 (\ud zg rg rg2 -> 0.9 * ud - 1e-4 * zg / sqrt(rg2 - rg ** 2 + 1e-4)) ugs gs rgup rg2up
--                -- ls2 = zipWith (\o g->o-g*0.1) ls gs
--                ls2 = force $ zipWith (+) ls ugup
--            learn3 sz ls2 rgup rg2up ugup l is2 os2 (gen + 1)
--            --    else return ls

--main1 :: IO ()
--main1 = do
----    n<-createRandomNetwork (RNNDimensions 1 2 3 True)
----    let (n2,out)=evalStep n [1]
----        (n3,out1)=evalStep n2 [3]
----        (n4,out2)=evalSteps n [[1],[3]]
----    print $ n3==n4
----    print $ out1 == (last out2)
----  txt <- liftM (T.take 100) $ T.readFile "data/tinyshakespeare.txt"
--  let -- dim =  (RNNDimensions 4 6 4 True)
--      --is  = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]
--      --os  = [[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
--      txt= "hello world!"
--      (is,os,m) = textToTrainData txt
--      sz = DM.size m
--      fn = "out.lstm.h"
--     --  dim = RNNDimensions sz (sz) sz True
--  print sz
--  fitnessList <- newIORef []
----  rnn1::LSTMNetwork <- read <$> readFile fn
--  ex <- doesFileExist fn
--  b <- if ex
--    then do
--        rnn1::LSTMNetwork <- read <$> readFile fn
--        return $ buildExisting rnn1 sz is os
--    else return $ build sz is os
--  r@(RNNData rnn _ _ _) <- runGAIO 64 0.1 b $ stopf fitnessList 100
----  -- (RNNData rnn _ _) <- evalRandIO $ buildNetworkData dim is os
----  print $ fitness r
--  -- print rnn
--  --let real = snd $ evalSteps rnn is
--  --print $ dataToText m real
--  t <- evalRandIO $ generate m (min 50 (T.length txt)) (size $ head is) rnn
--  print t
--  -- writeFile fn $ show rnn

-- build ::(Monad m,RandomGen g) =>  Int -> [Vector Double] -> [Vector Double] -> RandT g m (RNNData RNNetwork RNNDimensions)
-- build sz is os = buildNetworkData (RNNDimensions sz (sz) sz True) totalDataLength is os

--build ::(Monad m,RandomGen g) =>  TrainData a Int  -> RandT g m (RNNData LSTMNetwork Int)
--build sz td = buildNetworkData sz lstmFullSize is os

