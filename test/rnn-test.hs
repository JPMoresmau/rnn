{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
-- | Various tests
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
import Data.List
import Numeric

import Debug.Trace

main :: IO()
main = defaultMain tests

tests :: TestTree
tests = testGroup "Tests"
    [  testGroup "Utils" [
          testProperty "Matrix/vector product" prop_product
        , testCase "takes" $ do
            let l=[1,2,3,4,5,6]::[Int]
            [[1],[2,3],[4,5,6]] @=? takes ([1,2,3]::[Int]) l
            [[],[],[]] @=? takes [1,2,3] ([]::[Int])
        , testProperty "takes keep all data in order" prop_takes_concat
        , testProperty "takes has proper number of lists" prop_takes_length
        , testProperty "binary digits" prop_binary_digits
        , testCase "softmax" $ do
            let l=[1,2,3,4,5,6]::[Double]
                sf=softmax l
            sum sf @?= 1
        , testProperty "euclidian" prop_euclidian
        , testProperty "equilateral size" prop_equilateral_size
        , testProperty "equilateral distance" prop_equilateral_distance
        , testProperty "equilateral decoding" prop_equilateral_decoding
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
        , testCase "Text Sparse" $ do
            testTextDataS "hello"
            testTextDataS "hello world!"
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
            , testCase "pointMutation" $ testMutation pointMutation
            , testCase "swapMutation" $ testMutation swapMutation
            , testCase "insertMutation" $ testMutation insertMutation
        ]
      , testGroup "LSTM" [
            testCase "Check steps" checkLSTMSteps
          , testProperty "Property steps" prop_lstm_eval_steps
          , testCase "Check vector conversion" checkLSTMVector
          , testProperty "list and vector match" prop_lstm_step
          , testCase "Check list of LSTMs" checkLSTMList
          , testProperty "Property list" prop_lstm_list
        ]
       , testGroup "LSTMIO" [
            testCase "LSTMIO basics" checkLSTMIO
        ]
     ]
    ]

prop_product :: MatrixVector -> Bool
prop_product (MatrixVector sz ms vs) =
    let l=listMProd ms vs
        m=M.toList (M.matrix sz ms M.#> M.vector vs)
    in l == m -- (trace (show l) l)  == (trace (show m) m)

data MatrixVector = MatrixVector Int [Double] [Double]
    deriving (Show,Read,Eq,Ord)

instance Arbitrary MatrixVector where
    arbitrary = do
        Positive rows<-arbitrary
        Positive cols<-arbitrary
        ms <- vector (rows*cols)
        vs <- vector cols
        return $ MatrixVector cols ms vs

prop_takes_concat ::  String -> [Positive Int] -> Bool
prop_takes_concat xs ps = let
    idxs = map getPositive ps
    tot = sum idxs
    in concat (takes idxs xs) == take tot xs

prop_takes_length ::  String -> [Positive Int] -> Bool
prop_takes_length xs ps = let
    idxs = map getPositive ps
    in length (takes idxs xs) == length ps

prop_binary_digits :: Positive Int -> Bool
prop_binary_digits (Positive a) = binaryDigits a == length (showIntAtBase 2 intToDigit a "")


data EuclidianData = EuclidianData [Double] [Double]
    deriving Show

instance Arbitrary EuclidianData where
    arbitrary = do
        Positive sz<-arbitrary
        is1 <- vector sz
        is2 <- vector sz
        return $ EuclidianData is1 is2

prop_euclidian :: EuclidianData -> Bool
prop_euclidian (EuclidianData ds1 ds2) =
    euclidian (M.fromList ds1) (M.fromList ds2)
        == sum (map (\x->x*x) $ zipWith (-) ds2 ds1)

prop_equilateral_size :: Positive Int -> Bool
prop_equilateral_size (Positive n) =
    let m = equilateralEncoding n
    in (n,n-1) == (M.rows m,M.cols m)

prop_equilateral_distance :: Positive Int -> Bool
prop_equilateral_distance (Positive n) =
    let m = equilateralEncoding (n+1)
        ts = M.toRows m
        pairs = concatMap (\(x:ys)->map (\y->(x,y)) ys) $ init $ tails ts
        dis = map (roundTo 5 . uncurry euclidian) pairs
    in 1 == length (ordNub dis)

prop_equilateral_decoding :: Positive Int -> Bool
prop_equilateral_decoding (Positive n) =
    let rn = n + 1
        m  = equilateralEncoding rn
        vs= M.toRows m
        rs = map (equilateralDecoding m) vs
    in rs == [0..n]

testTextData :: T.Text -> IO()
testTextData t = do
    let (TrainData isSt is sz m) = textToTrainData t
    let charSet = T.foldl (flip DS.insert) DS.empty t
    DS.size charSet @=? sz+1
    t @=? dataToText m is
    T.init t @=? dataToText m (tail isSt)

testTextDataS :: T.Text -> IO()
testTextDataS t = do
    let (TrainData isSt is sz m) = textToTrainDataS t
    -- let charSet = T.foldl (flip DS.insert) DS.empty t
    -- sparseSize (DS.size charSet) @=? sz
    t @=? dataToTextS m is
    T.init t @=? dataToTextS m (tail isSt)

checkSteps :: Bool -> IO ()
checkSteps back = do
    (n::RNNetwork) <- evalRandIO $ randomNetwork (RNNDimensions 1 2 3 back) totalDataLength
    let (n2,out)=evalStep n $ M.fromList [1::Double]
        (n3,out1)=evalStep n2 $ M.fromList [3]
        (n4,out2)=evalSteps n [M.fromList [1],M.fromList [3]]
    n3 @=? n4
    out @=? head out2
    out1 @=? last out2


checkLSTMSteps :: IO ()
checkLSTMSteps = do
    (n::LSTMNetwork) <- evalRandIO $ randomNetwork 2 lstmFullSize
    let (n2,out)=evalStep n $ M.fromList [1::Double,2]
        (n3,out1)=evalStep n2 $ M.fromList [3,4]
        (n4,out2)=evalSteps n [M.fromList [1,2],M.fromList [3,4]]
    n3 @=? n4
    out @=? head out2
    out1 @=? last out2

prop_lstm_eval_steps :: LSTMData2 -> Bool
prop_lstm_eval_steps (LSTMData2 n1 is1 _ is2) =
    let
        (n2,out)=evalStep n1 $ M.fromList is1
        (n3,out1)=evalStep n2 $ M.fromList is2
        (n4,out2)=evalSteps n1 [M.fromList is1,M.fromList is2]
    in n3==n4 && out==head out2 && out1 == last out2

checkLSTMList :: IO ()
checkLSTMList = do
    (n1::LSTMNetwork) <- evalRandIO $ randomNetwork 2 lstmFullSize
    (n2::LSTMNetwork) <- evalRandIO $ randomNetwork 2 lstmFullSize
    let (n1_2,out)=evalStep n1 $ M.fromList [1::Double,2]
        (n2_2,out1)=evalStep n2 out
        (n3,out2)=evalStep [n1,n2] $ M.fromList [1,2]
    n3 @=? [n1_2,n2_2]
    out1 @=? out2

prop_lstm_list :: LSTMData2 -> Bool
prop_lstm_list (LSTMData2 n1 is1 n2 _) =
    let (n1_2,out)=evalStep n1 $ M.fromList is1
        (n2_2,out1)=evalStep n2 out
        (n3,out2)=evalStep [n1,n2] $ M.fromList is1
    in n3 == [n1_2,n2_2] && out1 == out2

checkLSTMVector :: IO ()
checkLSTMVector = do
    (n::LSTMNetwork) <- evalRandIO $ randomNetwork 2 lstmFullSize
    let arr = toVector n
        n2  = fromVector 2 arr
    n @=? n2

prop_lstm_step :: LSTMData -> Bool
prop_lstm_step (LSTMData n is)= snd (evalStep n (M.fromList is)) == M.fromList (snd $ lstmList 10 (M.toList $ toVector n) is)

checkLSTMIO :: IO()
checkLSTMIO = do
    print $ lstmFullSize 4
    print $ lstmioFullSize (2,4,3,1)
    (n1::LSTMIO) <- evalRandIO $ randomNetwork (2,4,3,1) lstmioFullSize
    let sz=rnnsize n1
        v1=toVector n1
        n2=fromVector sz v1
    n1 @=? n2
    let is=M.fromList [0.3,0.4]
        r1@(_,o1) = evalStep n1 is
    M.size o1 @?= 1
    let r2 = evalStep n2 is
    r1 @=? r2

data LSTMData = LSTMData LSTMNetwork [Double]
    deriving Show

instance Arbitrary LSTMData where
    arbitrary = do
        let sz=10
        ls <- vector (lstmFullSize sz)
        is <- vector sz
        let n = fromVector sz $ M.fromList ls
        return $ LSTMData n is


data LSTMData2 = LSTMData2 LSTMNetwork [Double] LSTMNetwork [Double]
    deriving Show

instance Arbitrary LSTMData2 where
    arbitrary = do
        let sz=10
        ls1 <- vector (lstmFullSize sz)
        is1 <- vector sz
        let n1 = fromVector sz $ M.fromList ls1
        ls2 <- vector (lstmFullSize sz)
        is2 <- vector sz
        let n2 = fromVector sz $ M.fromList ls2
        return $ LSTMData2 n1 is1 n2 is2

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
    notElem n1 rnns @? "n1 in result"
    notElem n2 rnns @? "n2 in result"

testMutation :: (LSTMNetwork -> Rand StdGen LSTMNetwork) -> IO()
testMutation f = do
    (n1::LSTMNetwork) <- evalRandIO $ randomNetwork 5 lstmFullSize
    n2 <- evalRandIO $ f n1
    toVector n1 /= toVector n2 @? "n1==n2"
