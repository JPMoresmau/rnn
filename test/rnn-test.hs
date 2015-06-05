{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where


import AI.Network.RNN.LSTM
import AI.Network.RNN.RNN
import AI.Network.RNN.Genetic
import AI.Network.RNN.Data
import AI.Network.RNN.Types

import Test.Tasty
import Test.Tasty.HUnit

import qualified Numeric.LinearAlgebra.HMatrix as M

import Control.Monad.Random
import qualified Data.Text as T

main :: IO()
main = defaultMain tests

tests :: TestTree
tests = testGroup "Tests"
    [ testGroup "RNN" [
        testCase "Check Steps without Back" $ checkSteps False
        , testCase "Check Steps with Back" $ checkSteps True
        , testCase "Check array conversion without Back" $ checkArray False
        , testCase "Check array conversion with Back" $ checkArray True
        , testCase "Check vector conversion without Back" $ checkArray False
        , testCase "Check vector conversion with Back" $ checkArray True
        ]
     , testGroup "Data" [
        testCase "Text" $ do
            testTextData "hello"
            testTextData "hello world!"
     , testGroup "Genetic" [
            testCase "MixVector" $ do
                let v1 = M.fromList [1,1,1,1]
                let v2 = M.fromList [2,2,2,2]
                (v3,v4) <- evalRandIO $ mixVector v1 v2 0.5
                (v3 /= v4) @? "v3 == v4"
                M.size v1 @=? M.size v3
                M.size v1 @=? M.size v4
            , testCase "crossNetworkFull" $ testCrossover crossNetworkFull
            , testCase "crossNetworkHalf" $ testCrossover crossNetworkHalf
        ]
      , testGroup "LSTM" [
            testCase "Check steps" $ checkLSTMSteps
          , testCase "Check vector conversion" $ checkLSTMVector
        ]
     ]
    ]

testTextData :: T.Text -> IO()
testTextData t = do
    let (is,_,m) = textToTrainData t
    t @=? dataToText m is

checkSteps :: Bool -> IO ()
checkSteps back = do
    (n::RNNetwork) <- evalRandIO $ randomNetwork (RNNDimensions 1 2 3 back) totalDataLength
    let (n2,out)=evalStep n $ M.fromList [1::Double]
        (n3,out1)=evalStep n2 $ M.fromList [3]
        (n4,out2)=evalSteps n $ [M.fromList [1],M.fromList [3]]
    n3 @=? n4
    out @=? (head out2)
    out1 @=? (last out2)


checkLSTMSteps :: IO ()
checkLSTMSteps = do
    (n::LSTMNetwork) <- evalRandIO $ randomNetwork 2 lstmFullSize
    let (n2,out)=evalStep n $ M.fromList [1::Double,2]
        (n3,out1)=evalStep n2 $ M.fromList [3,4]
        (n4,out2)=evalSteps n $ [M.fromList [1,2],M.fromList [3,4]]
    n3 @=? n4
    out @=? (head out2)
    out1 @=? (last out2)

checkLSTMVector :: IO ()
checkLSTMVector = do
    (n::LSTMNetwork) <- evalRandIO $ randomNetwork 2 lstmFullSize
    let arr = toVector n
        n2  = fromVector 2 arr
    n @=? n2


checkArray :: Bool -> IO ()
checkArray back = do
    let dim = RNNDimensions 1 2 3 back
    (n::RNNetwork) <- evalRandIO $ randomNetwork dim totalDataLength
    let arr = networkToArray n
        n2  = createNetworkFromArray dim arr
    case n2 of
        Right rn2 -> n @=? rn2
        Left err  -> assertFailure err

checkVector :: Bool -> IO ()
checkVector back = do
    let dim = RNNDimensions 1 2 3 back
    (n::RNNetwork) <- evalRandIO $ randomNetwork dim totalDataLength
    let arr = toVector n
        n2  = createNetworkFromVector dim arr
    case n2 of
        Right rn2 -> n @=? rn2
        Left err  -> assertFailure err


testCrossover :: (RNNetwork
                        -> RNNetwork -> Rand StdGen [RNNetwork])
                       -> IO ()
testCrossover f = do
    let dim = RNNDimensions 1 2 3 True
    (n1::RNNetwork) <- evalRandIO $ randomNetwork dim totalDataLength
    (n2::RNNetwork) <- evalRandIO $ randomNetwork dim totalDataLength
    rnns <- evalRandIO $ f n1 n2
    2 @=? length rnns
    (null $ filter (==n1) rnns) @? "n1 in result"
    (null $ filter (==n2) rnns) @? "n2 in result"

