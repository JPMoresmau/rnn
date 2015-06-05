{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import AI.Network.RNN
import AI.GeneticAlgorithm.Simple
import qualified Data.Map as DM
import Control.Monad
import Control.Monad.Random
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Numeric.LinearAlgebra.HMatrix (Vector)

main :: IO ()
main = do
--    n<-createRandomNetwork (RNNDimensions 1 2 3 True)
--    let (n2,out)=evalStep n [1]
--        (n3,out1)=evalStep n2 [3]
--        (n4,out2)=evalSteps n [[1],[3]]
--    print $ n3==n4
--    print $ out1 == (last out2)
  -- txt <- liftM (T.take 1000) $ T.readFile "data/tinyshakespeare.txt"
  let -- dim =  (RNNDimensions 4 6 4 True)
      --is  = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]
      --os  = [[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
      txt= "hello world"
      (is,os,m) = textToTrainData txt --"hello"
      sz = DM.size m
     --  dim = RNNDimensions sz (sz) sz True
  print sz
  r@(RNNData rnn _ _ _) <- runGAIO 64 0.1 (build sz is os) stopf
  -- (RNNData rnn _ _) <- evalRandIO $ buildNetworkData dim is os
  print $ fitness r
  -- print rnn
  --let real = snd $ evalSteps rnn is
  --print $ dataToText m real
  t <- evalRandIO $ generate m 50 (head is) rnn
  print t
  -- writeFile "out.rnn" $ show rnn

-- build ::(Monad m,RandomGen g) =>  Int -> [Vector Double] -> [Vector Double] -> RandT g m (RNNData RNNetwork RNNDimensions)
-- build sz is os = buildNetworkData (RNNDimensions sz (sz) sz True) totalDataLength is os

build ::(Monad m,RandomGen g) =>  Int -> [Vector Double] -> [Vector Double] -> RandT g m (RNNData LSTMNetwork Int)
build sz is os = buildNetworkData sz lstmFullSize is os
