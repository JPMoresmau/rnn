{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
module Main where


import AI.Network.RNN.LSTM
import AI.Network.RNN.RNN
import AI.Network.RNN.Genetic
import AI.Network.RNN.Data
import AI.Network.RNN.Types
import AI.Network.RNN.Util

import Test.Tasty
import Test.Tasty.HUnit
import Test.Tasty.QuickCheck

import qualified Numeric.LinearAlgebra.HMatrix as M

import Control.Monad.Random
import qualified Data.Text as T
import qualified Data.Set as DS

import Data.Char
import Numeric

main :: IO()
main = defaultMain tests

tests :: TestTree
tests = testGroup "Tests"
    [  testGroup "Utils" [
          testProperty "Matrix/vector product" prop_product
        , testCase "takes" $ do
            let l=[1,2,3,4,5,6]
            [[1],[2,3],[4,5,6]] @=? takes [1,2,3] l
            [[],[],[]] @=? takes [1,2,3] ([]::[Int])
        , testProperty "takes keep all data in order" prop_takes_concat
        , testProperty "takes has proper number of lists" prop_takes_length
        , testProperty "binary digits" prop_binary_digits
       ]
     , testGroup "RNN" [
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
        , testCase "TextBinary" $ do
            testTextDataB "hello"
            testTextDataB "hello world!"
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
          , testProperty "list and vector match" prop_lstm_step
        ]
     ]
    ]

prop_product :: MatrixVector -> Bool
prop_product (MatrixVector sz ms vs) = (listMProd ms vs) == (M.toList $ (M.matrix sz ms) M.#> (M.vector vs))

data MatrixVector = MatrixVector Int [Double] [Double]
    deriving (Show,Read,Eq,Ord)

instance Arbitrary MatrixVector where
    arbitrary = do
        Positive rows<-arbitrary
        Positive cols<-arbitrary
        ms <- vector (rows*cols)
        vs <- vector cols
        return $ MatrixVector cols ms vs

prop_takes_concat ::  [Char] -> [Positive Int] -> Bool
prop_takes_concat xs ps = let
    idxs = map getPositive ps
    tot = sum idxs
    in (concat $ takes idxs xs) == (take tot xs)

prop_takes_length ::  [Char] -> [Positive Int] -> Bool
prop_takes_length xs ps = let
    idxs = map getPositive ps
    in (length $ takes idxs xs) == (length ps)

prop_binary_digits :: Positive Int -> Bool
prop_binary_digits (Positive a) = (binaryDigits a) == (length $ showIntAtBase 2 intToDigit a "")

testTextData :: T.Text -> IO()
testTextData t = do
    let (TrainData isSt is sz m) = textToTrainData t
    let charSet = T.foldl (flip DS.insert) DS.empty t
    DS.size charSet @=? sz
    t @=? dataToText m is
    (T.init t) @=? (dataToText m $ tail isSt)

testTextDataB :: T.Text -> IO()
testTextDataB t = do
    let (TrainData isSt is sz m) = textToTrainDataB t
    let charSet = T.foldl (flip DS.insert) DS.empty t
    (binaryDigits $ DS.size charSet) @=? sz
    t @=? dataToTextB m is
    (T.init t) @=? (dataToTextB m $ tail isSt)

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

prop_lstm_step :: LSTMData -> Bool
prop_lstm_step (LSTMData n is)= (snd $ evalStep n (M.fromList is)) == (M.fromList $ snd $ lstmList 10 (M.toList $ toVector n) is)

data LSTMData = LSTMData LSTMNetwork [Double]
    deriving Show

instance Arbitrary LSTMData where
    arbitrary = do
        let sz=10
        ls <- vector (lstmFullSize sz)
        is <- vector sz
        let n = fromVector sz $ M.fromList ls
        return $ LSTMData n is

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

