{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
-- | Main entry point for executable, mainly for tests
module Main (main) where

import AI.Network.RNN
import AI.GeneticAlgorithm.Simple
import Control.Monad
import Control.Monad.Random hiding (fromList)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.IORef
import System.Directory
--import Debug.Trace
import System.Environment

import Data.Binary

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BSL

import Numeric.LinearAlgebra.HMatrix
import Numeric.AD



--main::IO()
--main = do
--    let m :: [Expr Double] = map (\n->Var ("m" ++ show n)) [1..400]
--        v :: [Expr Double] = map (\n->Var ("v" ++ show n)) [1..4]
--        mycost v m = sum $ listMProd m v
--        mygrad = grad (mycost (map auto v)) m
--        --mygrads = gradientDescent (mycost (map auto v))  m
--    print $ mygrad

main :: IO()
main = do
    txt <- liftM (T.take 100) $ T.readFile "data/tinyshakespeare.txt"
    let --txt= "hello world!"
        trainData = train txt
        fn = "output/lstm.one.100"
        generateLength = min 50 (T.length txt)
    print "Text:"
    print txt
    args <- getArgs
    when (length args<2) $ error "rnn gradient <maxgen> | rmsprop <maxgen> | generic <maxgen>"
    let fn2=fn++"."++head args
    ex <- doesFileExist fn2
    rnn<-case args of
        ("gradient":mg:_) -> do
            let maxGen = read mg
            rd <- readEx fn2 ex trainData
            gradient fn2 rd generateLength maxGen
        ("rmsprop":mg:_) -> do
            let maxGen = read mg
            rd <- readEx fn2 ex trainData
            rmsprop fn2 rd generateLength maxGen
        ("genetic":mg:_) -> do
            let maxGen = read mg
            b <- if ex
               then do
                 rnn1::LSTMNetwork <- fromVector (tdRecSize trainData) <$> decodeFile fn2
                 return $ buildExisting rnn1 trainData
               else return $ buildTD trainData
--            let b = buildTDs trainData
            genetic fn2 b generateLength maxGen
        _ -> error "rnn gradient <maxgen> | rmsprop <maxgen> | generic <maxgen>"
    encodeFile fn2 $ toVector rnn
    where
      readEx fn ex trainData =
          if ex
                then do
                 r::LSTMNetwork <- fromVector (tdRecSize trainData) <$> decodeFile fn
                 return $ RNNData r trainData (fromIntegral $ length $ tdInputs trainData)
                else
                  evalRandIO $ buildTD trainData
      (train, gener) = (textToTrainDataS,generateS)
      genStep maxGen = max 1 $ maxGen `div` 10
      gradient fn (RNNData r td _) generateLength maxGen = learnGradientDescent r td $ test fn generateLength maxGen
      rmsprop fn (RNNData r td _) generateLength maxGen = learnRMSProp r td $ test fn generateLength maxGen
      genetic fn r generateLength maxGen = do
        fitnessList <- newIORef []
        (RNNData rnn _ _) <- runGAIO 64 0.2 r $ stopf2 fn generateLength maxGen fitnessList
        return rnn
      test fn generateLength maxGen rnn td gen= do
        when (mod gen (genStep maxGen) == 0) $ do
            let c= cost rnn td
            print (show gen ++ ":" ++ show c)
            t <- evalRandIO $ gener (tdData td) generateLength (size $ head $ tdInputs td) rnn
            print t
            let fn2 = fn ++ "." ++ (show gen)
            let bs= BSL.toStrict $ encode $ toVector rnn
            BS.writeFile fn2 bs
        return $ gen < maxGen
      stopf2 fn generateLength maxGen fs rd@((RNNData rnn td _),_) gen = do
        _ <- test fn generateLength maxGen rnn td gen
        stopf fs maxGen rd gen

-- | Build LSTM network
buildTD ::(Monad m,RandomGen g) =>  TrainData a -> RandT g m (RNNData LSTMNetwork a)
buildTD td = buildNetworkData td (tdRecSize td) lstmFullSize

buildTDIO ::(Monad m,RandomGen g) =>  TrainData a -> RandT g m (RNNData LSTMIO a)
buildTDIO td =
    let layers = 2
        sz   = 40
     --   tds    = TrainData (tdInputs td) (tdOutputs td) (tdRecSize td,sz,layers,tdRecSize td) (tdData td)
     in buildNetworkData td (tdRecSize td,sz,layers,tdRecSize td) lstmioFullSize

-- | Build LSTM network
buildTDs ::(Monad m,RandomGen g) =>  TrainData a -> RandT g m (RNNData [LSTMNetwork] a)
buildTDs td =
    let layers = 2
     --   tds = TrainData (tdInputs td) (tdOutputs td) (replicate layers $ tdRecSize td) (tdData td)
    in  buildNetworkData td (replicate layers $ tdRecSize td)(sum . map lstmFullSize)


-- | Build from existing data for genetic algorithm, to restart from where we were
buildExisting :: (Monad m,RandomGen g) =>  LSTMNetwork -> TrainData a -> RandT g m (RNNData LSTMNetwork a)
buildExisting rnn1 td@(TrainData is _ _ _) = do
  -- g <- getSplit
  --let rnn2 = evalRand (mutateNetwork rnn1) g
  return $ RNNData rnn1 td (fromIntegral $ length is)

-- | Build from existing data for genetic algorithm, to restart from where we were
buildExistings :: (Monad m,RandomGen g) =>  [LSTMNetwork] -> TrainData a -> RandT g m (RNNData [LSTMNetwork] a)
buildExistings rnns td@(TrainData is _ _ _) = do
  -- g <- getSplit
  --let tds = TrainData (tdInputs td) (tdOutputs td) (replicate (length rnns) $ tdRecSize td) (tdData td)
  --let rnn2 = evalRand (mutateNetwork rnn1) g
  return $ RNNData rnns td (fromIntegral $ length is)

-- | Build from existing data for genetic algorithm, to restart from where we were
buildExistingIO :: (Monad m,RandomGen g) =>  LSTMIO -> TrainData a -> RandT g m (RNNData LSTMIO a)
buildExistingIO rnn1 td@(TrainData is _ _ _) = do
  -- g <- getSplit
  --let tds = TrainData (tdInputs td) (tdOutputs td) (rnnsize rnn1) (tdData td)
  --let rnn2 = evalRand (mutateNetwork rnn1) g
  return $ RNNData rnn1 td (fromIntegral $ length is)

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

